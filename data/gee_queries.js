// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/* Landsat 8 Surface Reflectance imagery */

// imports
var full_region = ee.FeatureCollection("users/yangsiyu007/Orinoquia_outline"),
    landsat_8_sr = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR");

/**
 * Function to mask clouds based on the pixel_qa band of Landsat 8 SR data.
 * @param {ee.Image} image input Landsat 8 SR image
 * @return {ee.Image} cloudmasked Landsat 8 image
 */
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}


/*  All months 2013, 2014. Surface reflectance  */

var sr_images = ee.ImageCollection((landsat_8_sr))
    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'pixel_qa'])
    // Filter to get only images in the rough region outline
    .filterBounds(full_region)
    // Filter to get images within the date range (first 1.5 years of Landsat 8)
    .filterDate('2013-01-01', '2014-12-31')
    // Sort by scene cloudiness, ascending.
    .sort('CLOUD_COVER', false)
    .map(maskL8sr)  // map is only available for ImageCollection; mosaic() makes into an Image
    .mosaic();

Map.addLayer(sr_images, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000, gamma: 1.4}, 'L8 SR');

// Export over full region
Export.image.toDrive({
  image: sr_images.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']),  // actually using SR product I can export all bands together
  description: 'wcs_orinoquia_sr_2013_2014',
  scale: 30,
  region: full_region,
  maxPixels: 651523504
});


/*  All months 2013, 2014. Surface reflectance, median composite  */

var sr_images = ee.ImageCollection((landsat_8_sr))
    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'pixel_qa'])
    // Filter to get only images in the rough region outline
    .filterBounds(full_region)
    // Filter to get images within the first three years of Landsat 8
    .filterDate('2013-01-01', '2014-12-31')
    // Sort by scene cloudiness, ascending.
    .sort('CLOUD_COVER', false)
    // map is only available for ImageCollection; mosaic() or a composite reducer makes into an Image
    .map(maskL8sr);

print(sr_images); // we can only export Image, not ImageCollection

var sr_images = sr_images.median();
print(sr_images);

Map.addLayer(sr_images, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000, gamma: 1.4}, 'L8 SR');

// Export over full region
Export.image.toDrive({
  image: sr_images,
  description: 'wcs_orinoquia_sr_median_2013_2014',
  scale: 30,
  region: full_region,
  maxPixels: 651523504
});


/*  All months from 2019-01-01 to 2020-04-25, Surface reflectance  */

var sr_images = ee.ImageCollection((landsat_8_sr))
    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'pixel_qa'])
    // Filter to get only images in the rough region outline
    .filterBounds(full_region)
    // Filter to get images within the first three years of Landsat 8
    .filterDate('2019-01-01', '2020-04-25')
    // Sort by scene cloudiness, ascending.
    .sort('CLOUD_COVER', false)
    .map(maskL8sr)  // map is only available for ImageCollection; mosaic() makes into an Image
    .mosaic();

Map.addLayer(sr_images, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000, gamma: 1.4}, 'L8 SR');

// Export over full region
Export.image.toDrive({
  image: sr_images.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']),
  description: 'wcs_orinoquia_sr_2019_202004',
  scale: 30,
  region: full_region,
  maxPixels: 651523504
});


/*  All months from 2019-01-01 to 2020-04-25, Surface reflectance, median composite  */

// masking first, then mosaic
var sr_images = ee.ImageCollection((landsat_8_sr))
    .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'pixel_qa'])
    // Filter to get only images in the rough region outline
    .filterBounds(full_region)
    // Filter to get images within the first three years of Landsat 8
    .filterDate('2019-01-01', '2020-04-25')
    // Sort by scene cloudiness, ascending.
    .sort('CLOUD_COVER', false)
    // map is only available for ImageCollection; mosaic() or a composite reducer makes into an Image
    .map(maskL8sr);

print(sr_images); // we can only export Image, not ImageCollection

var sr_images = sr_images.median();
print(sr_images);

Map.addLayer(sr_images, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000, gamma: 1.4}, 'L8 SR');

// Export over full region
Export.image.toDrive({
  image: sr_images,
  description: 'wcs_orinoquia_sr_median_2019_202004',
  scale: 30,
  region: full_region,
  maxPixels: 651523504
});




/* Elevation at 30m resolution from SRTM */

var srtm = ee.Image("USGS/SRTMGL1_003");  // SRTM Digital Elevation Data 30m

// to view
Map.addLayer(srtm, {min: 0, max: 400}, 'elevation');

var elevation_min_max = srtm.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: full_region.geometry(),
  scale: 30,
  maxPixels: 1e9
});

print(elevation_min_max);

var elevation_mean = srtm.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: full_region.geometry(),
  scale: 30,
  maxPixels: 1e9
});

print(elevation_mean);

var elevation_median = srtm.reduceRegion({
  reducer: ee.Reducer.median(),
  geometry: full_region.geometry(),
  scale: 30,
  maxPixels: 1e9
});

print(elevation_median);

var elevation_std_dev = srtm.reduceRegion({
  reducer: ee.Reducer.stdDev(),
  geometry: full_region.geometry(),
  scale: 30,
  maxPixels: 1e9
});
print(elevation_std_dev);

var elevation_hist = srtm.reduceRegion({
  reducer: ee.Reducer.histogram(),
  geometry: full_region.geometry(),
  scale: 30,
  maxPixels: 1e9
});
print(elevation_hist);

Export.image.toDrive({
  image: srtm,
  description: 'wcs_orinoquia_srtm',
  scale: 30,
  region: full_region,
  maxPixels: 651523504
});




/* Show outlines of regions */

// imports
var trial_region =
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-71.35726434159585, 4.100856324779771],
          [-71.35726434159585, 2.6641925300504146],
          [-69.89608270097085, 2.6641925300504146],
          [-69.89608270097085, 4.100856324779771]]], null, false),
    full_region = ee.FeatureCollection("users/yangsiyu007/Orinoquia_outline");

// Create an empty image into which to paint the features, cast to byte.
var empty = ee.Image().byte();

// Paint all the polygon edges with the same number and width, display.
var full_region_outline = empty.paint({
  featureCollection: full_region,
  color: 1,
  width: 3
});

Map.addLayer(full_region_outline, {palette: 'FF9900'}, 'full region outline');

var empty = ee.Image().byte();
var trial_region_outline = empty.paint({
  featureCollection: trial_region,
  color: 1,
  width: 3
});
Map.addLayer(trial_region_outline, {palette: '119900'}, 'trial region outline');

