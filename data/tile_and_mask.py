# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Tile scenes of imagery (Landsat 8 only) and corresponding polygon labels, and create masks for the latter
for training segmentation models. This uses the Solaris package and the conda env defined in `environment_solaris.yml`.

This is the script version of the notebook `data/3_tile_and_mask.ipynb`.
"""

import argparse
import os
import pickle
import sys
from datetime import datetime

import geopandas as gpd
import humanfriendly
from tqdm import tqdm

import solaris as sol

# these should be file names, not paths. TODO take basename
FILES_TO_EXCLUDE = []


def get_tif_paths(image_path):
    """ List the specified .tif imagery sources, excluding those in FILES_TO_EXCLUDE

    Args:
        image_path: Path to a single .tif file or a directory containing some

    Returns:
        A list of one or more .tif file paths
    """
    if os.path.isfile(image_path):
        if image_path.endswith('.tif'):
            print(f'file size: {humanfriendly.format_size(os.path.getsize(image_path))}')
            return [image_path]
        else:
            print('--image_path provided points to a file but is not a .tif file')
            sys.exit(1)

    if os.path.isdir(image_path):
        tif_paths = []
        for fn in os.listdir(image_path):
            if fn.endswith('.tif') and fn not in FILES_TO_EXCLUDE:
                tif_path = os.path.join(image_path, fn)
                print(f'file size: {humanfriendly.format_size(os.path.getsize(tif_path))}')
                tif_paths.append(tif_path)
        assert len(tif_paths) > 0, 'No .tif files found in the provided directory'
        return tif_paths


def get_lon_lat_from_tile_name(tile_name):
    """Returns _lon_lat"""
    parts = tile_name.split('_')
    lon_lat = f'_{parts[-2]}_{parts[-1].split(".tif")[0]}'
    return lon_lat


def main():
    parser = argparse.ArgumentParser(description='3 - tile and mask incoming data')
    parser.add_argument(
        '--image_path',
        required=True,
        help='Path to a TIFF file or a directory containing the relevant TIFF files - must end in .tif to be used'
    )
    parser.add_argument(
        '--label_path',
        required=True,
        help='The label polygons. Only .shp is supported currently'
    )
    parser.add_argument(
        '--label_property',
        required=True,
        help='The name of the property in each label polygon to use as the label. Property needs to be of int values.'
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help='This script will create the subdirectories tiles, tiles_labels and tiles_masks if they do not exist already'
    )
    parser.add_argument(
        '--tile_size',
        default=2000
    )
    parser.add_argument(
        '--region_outline',
        help='Only .shp supported currently.'
    )

    args = parser.parse_args()
    assert os.path.exists(args.image_path)
    assert os.path.exists(args.label_path)
    assert len(args.label_property) > 0
    assert args.tile_size > 0

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Do this first before tiling so we can make sure the label_property is on the shapes
    print('Loading and checking the vector labels...')
    start_time = datetime.now()

    landuse_shape = gpd.read_file(args.label_path)

    elapsed = datetime.now() - start_time

    print(f'Loaded vector labels in {humanfriendly.format_timespan(elapsed)}. Columns/properties found:')
    print(landuse_shape.dtypes)
    assert args.label_property in landuse_shape.dtypes, f'label_property {args.label_property} is not a column in the label dataframe'

    print('Exploding multi-polygons...')
    landuse_shape_exploded = landuse_shape.explode()
    print(f'Number of vector features provided: {len(landuse_shape)}.')
    print(f'Number of polygons after exploding {len(landuse_shape_exploded)}')

    print('Buffering the polygons... This may take several minutes')
    start_time = datetime.now()

    landuse_shape_exploded.geometry = landuse_shape_exploded.geometry.buffer(0)

    elapsed = datetime.now() - start_time
    print(f'Buffering took {humanfriendly.format_timespan(elapsed)}')

    # load region of interest polygon
    region_outline_poly = None
    if args.region_outline:
        # this is of type shapely.geometry.polygon.Polygon
        region_outline_poly = gpd.read_file(args.region_outline).loc[0, 'geometry']

    tif_paths = get_tif_paths(args.image_path)

    for tif_path in tif_paths:
        tif_name_no_extension = os.path.splitext(os.path.basename(tif_path))[0]

        print(f'\nProcessing tif {tif_name_no_extension}')

        print('Tiling images...')
        raster_tiler = sol.tile.raster_tile.RasterTiler(dest_dir=os.path.join(out_dir, 'tiles'),
                                                        # the directory to save images to
                                                        src_tile_size=(args.tile_size, args.tile_size),
                                                        aoi_boundary=region_outline_poly,
                                                        verbose=True)
        try:
            raster_tiler.tile(tif_path)
            # raster_tiler.tile_paths has the abs paths to the output small tifs
        except Exception as e:
            print('Exception in raster_tiler.tile()! ', str(e))

        # serialize the bounds as a record - TODO this is not used
        with open(os.path.join(out_dir, f'tile_bounds_{tif_name_no_extension}.pickle'), 'wb') as f:
            pickle.dump(raster_tiler.tile_bounds, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Tiling vector labels...')
        vector_tiler = sol.tile.vector_tile.VectorTiler(dest_dir=os.path.join(out_dir, 'tiles_labels'),
                                                        verbose=True)

        try:
            vector_tiler.tile(landuse_shape_exploded,
                              tile_bounds=raster_tiler.tile_bounds)
            # vector_tiler.tile_paths has the abs paths to the cut geojsons "geoms_*"
        except Exception as e:
            print('Exception in vector_tiler.tile()! ', str(e))

        print('Creating label masks...')
        os.makedirs(os.path.join(out_dir, 'tiles_masks'), exist_ok=True)

        for tile_path, tile_label_path in tqdm(zip(raster_tiler.tile_paths, vector_tiler.tile_paths)):
            try:
                lon_lat = get_lon_lat_from_tile_name(os.path.basename(tile_path))

                fp_mask = sol.vector.mask.footprint_mask(
                    df=tile_label_path,
                    out_file=os.path.join(out_dir, 'tiles_masks', 'mask{}.png'.format(lon_lat)),  # _ included
                    reference_im=tile_path,
                    burn_field=args.label_property)
            except Exception as e:
                print('Exception in footprint_mask()! ', str(e))
                continue
    print('Done.')


if __name__ == '__main__':
    print(f'solaris version: {sol.__version__}')
    main()
