# Data preparation

## Available data

- The labels WCS provided are a set of polygons each labeled with a land use type. They were created manually based on 2010-2012 Landsat data. WCS hopes to have a slightly different set of classes and updated land use maps based ideally on Sentinel data.

- Landsat 7 provides imagery for the period of 2010 to 2012, but its scan line corrector failure makes the image difficult to use. Creating composites is quite involved (although it may help with overfitting).

- Landsat 8 came online in 2018 so we have decided to use this instead. We use imagery from 2013 to 2014 inclusive.


## Steps

- Use convex hull (easier to implement than a concave hull but may have larger area with no labels) to get a polygon emcompassing the area that has land use labels.

- Use GEE with the outline polygon to get a list of the most cloudless scenes in the period of interest. Download these scenes either from GEE or its official source. To start with, download ~3 scenes to walk through the rest of the steps.

- Use Solaris to tile the scenes to the size required by the model and create multi-class label masks.


# Model output post-processing

## Polygonization

Consolidate the model output raster files:
```
gdalbuildvrt labels.vrt output_scenes/*.tif
```

```
gdalbuildvrt /home/otter/wcs/mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920.vrt /home/otter/wcs/mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920/res_wcs_orinoquia_sr_median_2019_202004*.tif
```

[Polygonize](https://gdal.org/programs/gdal_polygonize.html) the raster output as polygons in the shapefile format:

```
gdal_polygonize.py /home/otter/wcs/mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920.vrt -f "ESRI Shapefile" "/home/otter/wcs/mnt/wcs-orinoquia/delivered/20200715/results_coarse_baseline_201920.shp" "coarse_baseline_sr_median_2019_202004" "model_pred"
```
