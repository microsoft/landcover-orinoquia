# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This script performs post-processing steps for the model's output rasters. Currently this only involves applying
a majority filter within a radius specified as an input.

Should be run from the root of the repo so that the class list can be located in the `constants` folder.

Given an input directory containing the model's output in one or more TIF files, each will be processed
and saved to the output directory with "po_" prepended to the file name.

We can create a .vrt file after running this script, followed by the GDAL polygonization step.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
import skimage
from geospatial.visualization.raster_label_visualizer import RasterLabelVisualizer
from skimage.morphology import disk
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Post-processing of model output')
    parser.add_argument(
        '--model_output_dir',
        help='Path to a dir containing the model prediction patches/tiles as geoTIFFs.'
    )
    parser.add_argument(
        '--output_dir',
        default='./filtered_outputs',
        help='Path to a dir to put the processed patches/tiles.'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Radius used by the majority filter'
    )

    args = parser.parse_args()
    assert os.path.isdir(args.model_output_dir), 'Model output dir is not a directory'
    assert os.path.exists(args.model_output_dir), f'Model output dir is not found at {args.model_output_dir}'
    assert isinstance(args.radius, int) and args.radius > 0, 'Radius should be an integer > 0'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    viz = RasterLabelVisualizer('constants/class_lists/wcs_coarse_label_map.json')
    color_map = viz.get_tiff_colormap()

    neighborhood = disk(args.radius)

    for patch_name in tqdm(os.listdir(args.model_output_dir)):
        if Path(patch_name).suffix.lower() != '.tif':
            continue

        patch_reader = rasterio.open(os.path.join(args.model_output_dir, patch_name))
        patch = np.array(patch_reader.read()).squeeze()  # skimage expect 2D array

        print(f'Processing patch {patch_name}, shape is {patch.shape}')

        patch = skimage.filters.rank.majority(patch, neighborhood)
        patch = np.expand_dims(patch, axis=0)  # patch_writer expects a band dim in front

        patch_writer = rasterio.open(
                            os.path.join(args.output_dir, 'po_' + patch_name),
                            'w',
                            driver='GTiff',
                            height=patch_reader.height,
                            width=patch_reader.width,
                            count=1,
                            dtype=np.uint8,
                            crs=patch_reader.crs,
                            transform=patch_reader.transform,
                            nodata=0,
                            compress='lzw'
                        )
        patch_writer.write_colormap(1, color_map)  # the colormap is for band 1
        patch_writer.write(patch)
        patch_writer.close()  # save to disk


if __name__ == '__main__':
    main()
