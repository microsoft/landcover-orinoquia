"""
Execute this script in QGIS to add the queried Sentinel 2 imagery as an XYZ tile to the project.
"""


import ee
# import geopandas as gpd

# QGIS plug-in for GEE
from ee_plugin import Map

# import the region outline
# region_outline = gpd.read_file('/Users/siyuyang/Source/temp_data/WCS_land_use/outline/Orinoquia_outline.shp')
# region_outline_coords = list(region_outline.geometry[0].exterior.coords)  # geometry object to list of coords
# ee_region_outline = ee.Geometry.Polygon(region_outline_coords)

region_outline = ee.Geometry.Polygon([
 [-71.63069929490757, 8.096518229530101],
 [-67.04344372975483, 8.110020085498173],
 [-67.06369651370694, 4.221485566693199],
 [-71.63407475889959, 4.164102678828891],
 [-71.63069929490757, 8.096518229530101]])


# query for imagery
def mask_S2_clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)

sentinel2_aoi = ee.ImageCollection('COPERNICUS/S2_SR')\
    .select(['B2', 'B3', 'B4', 'QA60'])\
    .filterBounds(region_outline)

sentinel2_median_image = sentinel2_aoi.filterDate('2019-01-01', '2020-06-26')\
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
    .map(mask_S2_clouds)\
    .median()

rgb_vis = {
    'min': 0.0,
    'max': 3000,
    'gamma': 1.3,
    'bands': ['B4', 'B3', 'B2']}


Map.setCenter(-68.6345, 6.0289, 10)
Map.addLayer(sentinel2_median_image, rgb_vis, 'Sentinel2 2019 - 2020 June RGB')


# image = ee.Image('USGS/SRTMGL1_003')
# Map.addLayer(image, {'palette': ['black', 'white'], 'min': 0, 'max': 5000}, 'DEM')

