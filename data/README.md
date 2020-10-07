# Data preparation

## Available data

- The labels are a set of polygons each labeled with a land use type. They were created manually based ona variety of satellite imagery sources from 2010-2012, some of which were higher in resolution than Landsat.

- Landsat 7 provides imagery for this period, but its scan line corrector failure makes the image difficult to use. Numerous techniques exist to fill in the gaps, but the resultant imagery is very uneven in some areas in the region (although it may help with overfitting).

- Landsat 8 came online in 2018 so we have decided to use it instead. We use imagery from 2013 to 2014 inclusive for training. The mismatch in imagery used during labeling and training is a source of noise. A median composite was created after masking each available scene in the two-year period. 

- The updated land cover map was based on a median composite created the same way using data from 2019 to April 2020.


## Steps

- Manually drew a rough outline of the region in QGIS. 

- In Google Earth Engine (GEE), create the median composites and download them ([gee_queries.js](./gee_queries.js)). Also download the SRTM DEM data for the region.

- Use Solaris to tile the scenes to the size required by the model and create multi-class label masks ([tile_and_mask.py](./tile_and_mask.py)).

- Append the DEM data as an extra channel to the tiles (to use the interactive fine-tuning tool, all data needed for inference need to be in the tiles).

- To speed up data reading during training, we also make shards of chips in the model's input size ([make_chip_shards.py](./make_chip_shards.py)), using a particular experiment's configuration file, which has functions to produce the chips (examples in [training_wcs/experiments](../training_wcs/experiments)).


# Model output post-processing

## Polygonization

1. Consolidate the model output raster files:
    ```
    gdalbuildvrt predictions.vrt output_scenes/*.tif
    ```
    
    ```
    gdalbuildvrt /mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920.vrt /mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920/res_wcs_orinoquia_sr_median_2019_202004*.tif
    ```
    
    GDAL command-line tools should be installed in the conda environment.

2. [Polygonize](https://gdal.org/programs/gdal_polygonize.html) the raster output as polygons in the shapefile format:

    ```
    gdal_polygonize.py /mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920.vrt -f "ESRI Shapefile" "/mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920.shp" "coarse_baseline_sr_median_2019_202004" "model_pred"
    ```
3. Get rid of polygons with area below the Minimum Mapping Area (MMA) that WCS decided on based on statistics of the raw model output.
