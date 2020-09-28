"""
First experiment using 6 channels (2, 3, 6, 7, NDVI, elevation) with the 13 + 1 coarse categories
mapped June 29 2020.
"""

import os
import sys

import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ai4eutils needs to be on the PYTHONPATH
from geospatial.enums import ExperimentConfigMode
from geospatial.visualization.imagery_visualizer import ImageryVisualizer
from geospatial.visualization.raster_label_visualizer import RasterLabelVisualizer

from training_wcs.scripts.models.unet.unet import Unet
from training_wcs.scripts.utils.data_transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip
from training_wcs.scripts.utils.datasets import SingleShardChipsDataset

experiment_name = 'wcs_coarse_baseline'

config_mode = ExperimentConfigMode.SCORING

# I/O -------------------------------------------------------------------------------------------------
if config_mode in [ExperimentConfigMode.PREPROCESSING, ExperimentConfigMode.TRAINING]:
    data_shard_dir = '/boto_disk_0/wcs_data/shards/full_sr_median_2013_2014_elevation'

    # a dir with experiment_name will be created in here and checkpoints are saved here
    # set as './outputs' for AML to stream to this Run's folder
    out_dir = f'/boto_disk_0/wcs/{experiment_name}/outputs'
    os.makedirs(out_dir, exist_ok=True)

    # TF events go here. Set it as './logs' if using AML so they can be streamed
    log_dir = f'/boto_disk_0/wcs/{experiment_name}/logs'  # './logs'
    os.makedirs(log_dir, exist_ok=True)

# for scoring script and make_chip_shards
if config_mode in [ExperimentConfigMode.PREPROCESSING, ExperimentConfigMode.SCORING]:
    prediction_window_size = 128

    label_viz = RasterLabelVisualizer(
        label_map='/home/boto/wcs/pycharm/constants/class_lists/wcs_coarse_label_map.json')

    data_dir = '/boto_disk_0/wcs_data'  # which contains images_srtm

# Training ----------------------------------------------------------------------------------------------

# this is the *total* epoch; if restarting from a checkpoint, be sure to add the additional number of epochs
# to fine-tune on top of the original value of this var
total_epochs = 500
print_every = 150  # print every how many steps; just the minibatch loss and accuracy
assert print_every >= 1, 'print_every needs to be greater than or equal 1'

starting_checkpoint_path = None

init_learning_rate = 5e-5

batch_size = 28

# visualizing results on a sample of chips during training_wcs
num_chips_to_viz = 48

# Hardware and framework --------------------------------------------------------------------------------
dtype = torch.float32

# Model -------------------------------------------------------------------------------------------------

num_classes = 14  # empty plus the 13 *coarse* WCS classes; this is the number of output nodes

num_in_channels = 6  # 2, 3, 6, 7, NDVI, elevation

# the smallest number of filters is 64 when feature_scale is 1, and it is 32 when feature_scale is 2
feature_scale = 1

is_deconv = True  # True to use transpose convolution filters to learn upsampling; otherwise upsampling is not learnt

is_batchnorm = True

model = Unet(feature_scale=feature_scale,
             n_classes=num_classes,
             in_channels=num_in_channels,
             is_deconv=is_deconv,
             is_batchnorm=is_batchnorm)

# Data ---------------------------------------------------------------------------------------------------

common_classes = [
]
less_common_classes = [
]

weights = []
for i in range(num_classes):
    if i in common_classes:
        weights.append(0)
    elif i in less_common_classes:
        weights.append(0)
    else:
        weights.append(1)

loss_weights = torch.FloatTensor(weights)  # None if no weighting for classes
print('Weights on loss per class used:')
print(loss_weights)

# how many subprocesses to use for data loading
# None for now - need to modify datasets.py to use
data_loader_num_workers = None

# not available in IterableDataset data_loader_shuffle = True  # True to have the data reshuffled at every epoch

chip_size = 256

# datasets and dataloaders
if config_mode == ExperimentConfigMode.PREPROCESSING:
    dset_train = SingleShardChipsDataset(data_shard_dir, shard_prefix='train', channels=None,
                                         transform=transforms.Compose([
                                             ToTensor(),
                                             RandomHorizontalFlip(),  # these operate on Tensors, not PIL images
                                             RandomVerticalFlip()
                                         ]))
    loader_train = DataLoader(dset_train,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=True)  # currently num_workers is None

    dset_val = SingleShardChipsDataset(data_shard_dir, shard_prefix='val', channels=None,
                                       transform=transforms.Compose([
                                           ToTensor(),
                                           RandomHorizontalFlip(),  # these operate on Tensors, not PIL images
                                           RandomVerticalFlip()
                                       ]))
    loader_val = DataLoader(dset_val,
                            num_workers=4,
                            batch_size=batch_size)

# Data shards generation configurations --------------------------------------------------------------------

# These configurations are copied from training_wcs/experiments/elevation/elevation_2_config.py
# They are only used with make_chip_shards.py and infer.py
# train.py only use the generated chip shards as numpy arrays
if config_mode in [ExperimentConfigMode.PREPROCESSING, ExperimentConfigMode.SCORING]:
    elevation_path = os.path.join(data_dir, 'images_srtm', 'wcs_orinoquia_srtm.tif')
    elevation_reader = rasterio.open(elevation_path)

# based on min and max values from the sample tile
# wcs_orinoquia_sr_median_2013_2014-0000007424-0000007424_-71.347_4.593.tif in training_wcs set
# bands 4 and 5 are combined to get the NDVI, so the normalization params for 4 and 5 are
# not used during training_wcs data generation, only for visualization (actually not yet used for viz either).
bands_normalization_params = {
    # these are the min and max to clip to for the band
    'min': {
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0
    },
    'max': {
        2: 700,
        3: 1500,
        4: 1500,
        5: 5000,
        6: 5000,
        7: 3000
    },
    'gamma': {  # all the same in this experiment with value 1 which means no effect
        2: 1.0,
        3: 1.0,
        4: 1.0,
        5: 1.0,
        6: 1.0,
        7: 1.0
    }
}

elevation_standardization_params = {
    # from calculations done in GEE
    'mean': 399.78,
    'std_dev': 714.78
}


def get_elevation_chip(tile_reader, chip_window):
    x, y = (tile_reader.bounds.left, tile_reader.bounds.top)

    # getting the pixel array indices corresponding to points in georeferenced space
    row, col = elevation_reader.index(x, y)

    # tile wcs_orinoquia_sr_median_2013_2014-0000000000-0000007424_-72.425_7.671.tif
    # top left corner looks up to a negative row index. Clipping to 0 seems to be okay visually
    row = max(0, row)
    col = max(0, col)

    # resolution and project are the same for the elevation data and the Landsat imagery
    row = row + chip_window[1]
    col = col + chip_window[0]  # x is col

    try:
        w = rasterio.windows.Window.from_slices((row, row + chip_window[3]), (col, col + chip_window[2]))
    except Exception as e:
        print(str(e))
        print('chip window:', str(chip_window))
        print('original row and col: ', str(elevation_reader.index(x, y)))
        print('row:', row)
        print('col:', col)
        print(tile_reader.bounds)
        print('x:', x)
        print('y:', y)
        import sys
        sys.exit(1)

    chip_elevation = elevation_reader.read(1, window=w)  # only 1 band

    # standardize
    chip_elevation = (chip_elevation - elevation_standardization_params['mean']) / elevation_standardization_params[
        'std_dev']
    return chip_elevation


def _pad_chip(band, chip_window):
    """

    Args:
        band: numpy array of dims (h, w)
        chip_window: (col_off x, row_off y, width, height)

    Returns:
        band padded to dims (width, height) of the chip_window provided
    """
    width = chip_window[2]
    height = chip_window[3]
    # check for smaller than because we use get_chip to get the entire tile during scoring
    if band.shape[0] < height or band.shape[1] < width:
        # default pad constant value is 0
        try:
            band = np.pad(band,
                          [(0, height - band.shape[0]), (0, width - band.shape[1])])
        except Exception as e:
            print(f'coarse_baseline_config, _pad_chip exception: {e}')

            sys.exit(1)
    return band


normal_bands = [2, 3, 6, 7]  # bands to be used without calculating other indices e.g. NDVI


def get_chip(tile_reader, chip_window, chip_for_display=True):
    """
    Get an area (chip) specified by the chip_window. Is not related to chip_size
    Args:
        tile_reader: rasterio dataset object of the imagery tile
        chip_window: (col_off x, row_off y, width, height)
        chip_for_display: True if also return a chip that looks good
    Returns:
        stacked: A numpy array of dims (6, H, W) - note that height and width are switched from chip_window
        chip_for_display: If chip_for_display is True, also a 3-band array of the RGB channels scaled to
            look good (R channel not included in stacked)
    """

    bands_to_stack = []
    for b in normal_bands:
        # getting one band at a time because min, max and gamma may be different
        band = ImageryVisualizer.show_landsat8_patch(
            tile_reader,
            bands=[b],  # pass in a list to get the batch dimension in the results
            window=chip_window,
            band_min=bands_normalization_params['min'][b],
            band_max=bands_normalization_params['max'][b],
            gamma=bands_normalization_params['gamma'][b],
            return_array=True
        )

        # deal with incomplete chips
        band = _pad_chip(band, chip_window)

        bands_to_stack.append(band)  # band is 2D (h, w), already squeezed, dtype is float 32

    ndvi = ImageryVisualizer.get_landsat8_ndvi(tile_reader, window=chip_window)  # 2D, dtype is float32
    ndvi = _pad_chip(ndvi, chip_window)
    bands_to_stack.append(ndvi)

    elevation = get_elevation_chip(tile_reader, chip_window)  # scene covers entire region, not tiled, so no gaps
    elevation = _pad_chip(elevation, chip_window)
    bands_to_stack.append(elevation)

    try:
        stacked = np.stack(bands_to_stack)
    except Exception as e:
        print(f'Exception in get_chip: {e}')
        for b in bands_to_stack:
            print(b.shape)
        print('')

    assert stacked.shape == (6, chip_window[3], chip_window[2]), \
        f'Chip has wrong shape: {stacked.shape}, should be (6, h, w)'

    if chip_for_display:
        # chip for display, getting the RBG bands (default) with a different gamma and band_max that look good
        chip_for_display = ImageryVisualizer.show_landsat8_patch(tile_reader,
                                                                 window=chip_window,
                                                                 band_max=3000,  # what looks good for RGB
                                                                 gamma=0.5,
                                                                 return_array=True)

        return stacked, chip_for_display
    else:
        return stacked


def preprocess_tile(tile_array: np.ndarray) -> np.ndarray:
    """Same functionality as get_chip(), but applies to a numpy array tile of arbitrary shape.
    Currently only used with the landcover tool.

    Args:
        tile_array: A numpy array of dims (height, width, channels). Expect elevation to be the eleventh channel

    Returns:
        Numpy array representing of the preprocessed chip of dims (6, height, width) - note that channels is
        in-front.
    """
    bands_to_stack = []
    for b in normal_bands:
        # getting one band at a time because min, max and gamma may be different
        band = ImageryVisualizer.show_landsat8_patch(
            tile_array,
            bands=[b],  # pass in a list to get the batch dimension in the results
            band_min=bands_normalization_params['min'][b],
            band_max=bands_normalization_params['max'][b],
            gamma=bands_normalization_params['gamma'][b],
            return_array=True
        )
        bands_to_stack.append(band)  # band is 2D (h, w), already squeezed, dtype is float 32

    ndvi = ImageryVisualizer.get_landsat8_ndvi(tile_array)  # 2D, dtype is float32
    bands_to_stack.append(ndvi)

    # for the interactive tool, elevation is band 11 (1-indexed) or 10 (0-indexed), and already normalized
    # by elevation_standardization_params (could have done the normalization here too)
    elevation = tile_array[:, :, 10]
    bands_to_stack.append(elevation)

    stacked = np.stack(bands_to_stack)

    assert stacked.shape == (
    6, tile_array.shape[0], tile_array.shape[1]), f'preprocess_tile, wrong shape: {stacked.shape}'
    return stacked
