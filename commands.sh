# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#@IgnoreInspection BashAddShebang

# A record of often-used bash commands.


# Add ai4eutils to PYTHONPATH

export PYTHONPATH=${PYTHONPATH}:/home/<username>/wcs/repos/ai4eutils



# Tiling and masking the 2013-2014 set

python data/3_tile_and_mask.py \
--image_path /home/<username>/wcs/mnt/wcs-orinoquia/images_sr_median/2013_2014  \
--label_path /home/<username>/wcs/mnt/wcs-orinoquia/provided_labels/Landuse_shape/derived_20200421/landuse.shp \
--label_property Landuse_WC \
--out_dir /home/<username>/wcs/mnt/wcs-orinoquia/tiles/full_sr_median_2013_2014 \
--region_outline /home/<username>/wcs/mnt/wcs-orinoquia/misc/outline/Orinoquia_outline.shp



# Training locally on a DSVM

conda activate wcs

export PYTHONPATH="${PYTHONPATH}:/home/<username>/wcs/pycharm"

cd training

sh run_training_local.sh



# Scoring

# modify selection of tiles at the top of inference.py

checkpoint_path=/disk/wcs/wcs_coarse_baseline_0_wrong_val_viz/outputs/wcs_coarse_baseline/checkpoints/model_best.pth.tar

out_dir=/home/<username>/wcs/mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920

python training/inference.py --config_module_path training/experiments/coarse_baseline/coarse_baseline_config_refactored.py --checkpoint_path ${checkpoint_path} --out_dir ${out_dir} --output_softmax


# Post-processing

python data/postprocess_model_output.py --model_output_dir .../Data/WCSColombia/delivered/20200715/results_coarse_baseline_201920 --output_dir .../Data/WCSColombia/delivered/20200715/results_coarse_baseline_201920_polygons_filtered7 --radius 7


# Combine output raster tiles into a .vrt

gdalbuildvrt .../Data/WCSColombia/delivered/20200715/results_coarse_baseline_201920_polygons_filtered7/po_res_wcs_orinoquia_sr_median_2019_202004.vrt .../Data/WCSColombia/delivered/20200715/results_coarse_baseline_201920_polygons_filtered7/*.tif


# Mounting blob container

sh mount_wcs_containers.sh

OR

sudo mkdir /mnt/blobfusetmp-wcs-orinoquia
sudo chown USERNAME /mnt/blobfusetmp-wcs-orinoquia

blobfuse /home/<usrname>/wcs/mnt/wcs-orinoquia --tmp-path=/mnt/blobfusetmp-wcs-orinoquia --config-file=wcs-orinoquia.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other


# Copy data to disk for faster access

azcopy cp "https://geospatialmldata.blob.core.windows.net/wcs-orinoquia/tiles/full_sr_median_2013_2014/tiles?SAS_KEY" /disk/wcs_data/tiles/full_sr_median_2013_2014 --recursive

azcopy cp "https://geospatialmldata.blob.core.windows.net/wcs-orinoquia/tiles/full_sr_median_2013_2014/tiles_masks?SAS_KEY" /disk/wcs_data/tiles/full_sr_median_2013_2014 --recursive

azcopy cp "https://geospatialmldata.blob.core.windows.net/wcs-orinoquia/tiles/full_sr_median_2013_2014/tiles_masks_coarse?SAS_KEY" /disk/wcs_data/tiles/full_sr_median_2013_2014 --recursive


Elevation data:

azcopy cp "https://geospatialmldata.blob.core.windows.net/wcs-orinoquia/images_srtm?SAS_KEY" /disk/wcs_data --recursive
