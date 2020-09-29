# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Scores a list of tiles specified at the top of this script.

Scoring is not batched i.e. batch size is 1. Some squeeze() need to be rid of to use for larger batch sizes.
"""

import argparse
import importlib
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import rasterio
import torch

# SPECIFY input_tiles as a list of absolute paths to imagery .tif files

# with open('/home/boto/wcs/pycharm/constants/splits/full_sr_median_2013_2014_splits.json') as f:
#     input_tiles_json = json.load(f)
# val_tiles = input_tiles_json['val']
# input_tiles = [os.path.join('/boto_disk_0/wcs_data/tiles/full_sr_median_2013_2014/tiles', i) for i in
#                val_tiles]

# all 2019 - 2020 tiles
tiles_dir = '/home/boto/wcs/mnt/wcs-orinoquia/images_sr_median/2019_202004'
input_tiles = [os.path.join(tiles_dir, i) for i in os.listdir(tiles_dir)]


def write_colormap(tile_writer, config):
    """Write the experiment config's RasterLabelVisualizer's colormap to a TIFF file"""
    tile_writer.write_colormap(1, config.label_viz.get_tiff_colormap())  # 1 for band 1


def main():
    parser = argparse.ArgumentParser(description='Inference on tiles')
    parser.add_argument(
        '--config_module_path',
        required=True,
        help="Path to the .py file containing the experiment's configurations"
    )
    parser.add_argument(
        '--checkpoint_path',
        required=True,
        help='Path to the checkpoint .tar file to use for scoring'
    )
    parser.add_argument(
        '--out_dir',
        default='./outputs',
        help='Path to a dir to put the prediction tiles.'
    )
    parser.add_argument(
        '--output_softmax',
        action='store_true',
        help='Outputs the softmax probability scores as tiff tiles too, and its visualization as RGB tiles.'
    )
    args = parser.parse_args()

    assert os.path.exists(args.checkpoint_path), f'Checkpoint at {args.checkpoint_path} does not exist.'

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # config for the training_wcs run that generated the model checkpoint
    try:
        module_name = 'config'
        spec = importlib.util.spec_from_file_location(module_name, args.config_module_path)
        config = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = config
        spec.loader.exec_module(config)
    except Exception as e:
        print(f'Failed to import the configurations. Exception: {e}')
        sys.exit(1)

    # device configuration
    print(f'Using PyTorch version {torch.__version__}.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}.')

    # scoring configuration
    chip_size = config.chip_size
    prediction_window_size = config.prediction_window_size if config.prediction_window_size else 128
    prediction_window_offset = int((chip_size - prediction_window_size) / 2)
    print((f'Using chip_size {chip_size} and window_size {prediction_window_size}. '
           f'So window_offset is {prediction_window_offset}'))

    start_time = datetime.now()

    # instantiate the model and then load its state from the given checkpoint
    model = config.model

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Using checkpoint at epoch {checkpoint['epoch']}, step {checkpoint['step']}, "
          f"val accuracy is {checkpoint.get('val_acc', 'Not Available')}")

    model = model.to(device=device)
    model.eval()  # eval mode: norm or dropout layers will work in eval mode instead of training_wcs mode

    with torch.no_grad():  # with autograd engine deactivated
        for i_file, input_tile_path in enumerate(input_tiles):

            out_path_hardmax = os.path.join(out_dir, 'res_' + os.path.basename(input_tile_path))
            if os.path.exists(out_path_hardmax):
                print(f'Skipping already scored tile {out_path_hardmax}')
                continue

            print(f'Scoring input tile {i_file} out of {len(input_tiles)}, {input_tile_path}')

            # dict_scores = {}  # dict of window tuple to numpy array of scores

            # load entire tile into memory
            tile_reader = rasterio.open(input_tile_path)

            # use the get_chip function (to normalize the bands in a way that's consistent with training_wcs)
            # but get a chip that's the size of the tile - all at once
            whole_tile_window = (0, 0, tile_reader.width, tile_reader.height)
            data_array: np.ndarray = config.get_chip(tile_reader, whole_tile_window, chip_for_display=False)

            # pad by mirroring at the edges to facilitate predicting only on the center crop
            data_array = np.pad(data_array,
                                [
                                    (0, 0),  # only pad height and width
                                    (prediction_window_offset, prediction_window_offset),  # height / rows
                                    (prediction_window_offset, prediction_window_offset)  # width / cols
                                ],
                                mode='symmetric')

            # set up hardmax prediction output tile
            tile_writer_hardmax = rasterio.open(
                out_path_hardmax,
                'w',
                driver='GTiff',
                height=tile_reader.height,
                width=tile_reader.width,
                count=1,  # only 1 "band", the hardmax predicted label at the pixel
                dtype=np.uint8,
                crs=tile_reader.crs,
                transform=tile_reader.transform,
                nodata=0,
                compress='lzw',
                blockxsize=prediction_window_size,
                # reads and writes are most efficient when the windows match the dataset’s own block structure
                blockysize=prediction_window_size
            )
            write_colormap(tile_writer_hardmax, config)

            # set up the softmax output tile
            if args.output_softmax:
                out_path_softmax = os.path.join(out_dir, 'prob_' + os.path.basename(input_tile_path))
                # probabilities projected into RGB for intuitive viewing
                out_path_softmax_viz = os.path.join(out_dir, 'prob_viz_' + os.path.basename(input_tile_path))

                tile_writer_softmax = rasterio.open(
                    out_path_softmax,
                    'w',
                    driver='GTiff',
                    height=tile_reader.height,
                    width=tile_reader.width,
                    count=config.num_classes,  # as many "bands" as there are classes to house the softmax probabilities
                    # quantize probabilities scores so each can be stored as one byte instead of 4-byte float32
                    dtype=np.uint8,
                    crs=tile_reader.crs,
                    transform=tile_reader.transform,
                    nodata=0,
                    compress='lzw',
                    blockxsize=prediction_window_size,
                    blockysize=prediction_window_size
                )
                tile_writer_softmax_viz = rasterio.open(
                    out_path_softmax_viz,
                    'w',
                    driver='GTiff',
                    height=tile_reader.height,
                    width=tile_reader.width,
                    count=3,  # RGB
                    dtype=np.uint8,
                    crs=tile_reader.crs,
                    transform=tile_reader.transform,
                    nodata=0,
                    compress='lzw',
                    blockxsize=prediction_window_size,
                    blockysize=prediction_window_size
                )

            # score the tile in windows
            num_rows = math.ceil(tile_reader.height / prediction_window_size)
            num_cols = math.ceil(tile_reader.width / prediction_window_size)

            for col_idx in range(num_cols):

                col_start = col_idx * prediction_window_size
                col_end = col_start + chip_size

                for row_idx in range(num_rows):

                    row_start = row_idx * prediction_window_size
                    row_end = row_start + chip_size

                    chip = data_array[:, row_start:row_end, col_start: col_end]

                    # pad to (chip_size, chip_size)
                    chip = np.pad(chip,
                                  [(0, 0), (0, chip_size - chip.shape[1]), (0, chip_size - chip.shape[2])])

                    # processing it as the dataset loader _get_chip does
                    chip = np.nan_to_num(chip, nan=0.0, posinf=1.0, neginf=-1.0)
                    sat_mask = chip[0].squeeze() > 0.0  # mask out DEM data where there's no satellite data
                    chip = chip * sat_mask

                    chip = np.expand_dims(chip, axis=0)
                    chip = torch.FloatTensor(chip).to(device=device)

                    try:
                        scores = model(chip)  # these are scores before the final softmax
                    except Exception as e:
                        print(f'Exception in scoring loop model() application: {e}')
                        print(f'Chip has shape {chip.shape}')
                        sys.exit(1)

                    _, preds = scores.max(1)
                    softmax_scores = torch.nn.functional.softmax(scores, dim=1)

                    softmax_scores = softmax_scores.cpu().numpy()  # (batch_size, num_classes, H, W)
                    preds = preds.cpu().numpy().astype(np.uint8)

                    assert np.max(preds) < config.num_classes

                    # model output needs to be cropped to the window so they can be written correctly into the tile

                    # same order as rasterio window: (col_off x, row_off y, width delta_x, height delta_y)
                    valid_window_tup = (
                        col_start,
                        row_start,
                        min(prediction_window_size, tile_reader.width - col_idx * prediction_window_size),
                        min(prediction_window_size, tile_reader.height - row_idx * prediction_window_size)
                    )

                    # preds has a batch dim here
                    preds1 = preds[:,
                             prediction_window_offset:prediction_window_offset + valid_window_tup[3],
                             prediction_window_offset:prediction_window_offset + valid_window_tup[
                                 2]]  # last dim is the inner most, x, width

                    # debug - print(f'col is {col_idx}, row is {row_idx}, valid_window_tup is {valid_window_tup}, preds shape: {preds.shape}, preds1 shape: {preds1.shape}')

                    window = rasterio.windows.Window(valid_window_tup[0], valid_window_tup[1],
                                                     valid_window_tup[2], valid_window_tup[3])
                    tile_writer_hardmax.write(preds1, window=window)

                    if args.output_softmax:
                        # cropping, e.g. from (1, 14, 256, 256) to (1, 14, 128, 128)
                        softmax_scores = softmax_scores[:, :,
                                         prediction_window_offset:prediction_window_offset + valid_window_tup[3],
                                         prediction_window_offset:prediction_window_offset + valid_window_tup[2]]

                        # get rid of batch dim. First dim for TIFF writer needs to be number of bands to write
                        softmax_scores = softmax_scores.squeeze()

                        # quantize using 256 buckets so probability value is writen in 1 byte instead of 4 (float32)
                        softmax_scores_quantized = (softmax_scores * 255).astype(np.uint8)

                        tile_writer_softmax.write(softmax_scores_quantized, window=window)

                        # softmax_scores_project is of dims ((batch_size), H, W, 3)
                        # make sure to not use the quantized version!
                        softmax_scores_proj = config.label_viz.visualize_softmax_predictions(softmax_scores)

                        softmax_scores_proj = softmax_scores_proj.squeeze()  # assume batch_size is 1 TODO

                        # we need it to be (3, H, W) to write to TIFF
                        softmax_scores_proj = np.transpose(softmax_scores_proj, axes=(2, 0, 1)).astype(np.uint8)
                        # debug - print(f'softmax_scores_proj min is {np.min(softmax_scores_proj)}, max is {np.max(softmax_scores_proj)}')

                        tile_writer_softmax_viz.write(softmax_scores_proj, window=window)

            tile_writer_hardmax.close()
            tile_writer_softmax.close()
            tile_writer_softmax_viz.close()

    duration = datetime.now() - start_time
    print(f'Inference finished in {duration}, which is {duration / len(input_tiles)} per tile')


if __name__ == '__main__':
    main()
