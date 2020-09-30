# Model training


## Data

### Batching
Each tile is chipped in order (x then y, finishing one row then move to the next row of chips) before moving to the next tile.  

### Folder structure and splits

The splits are stored in a JSON file, as a dict with `train` and `val` fields, pointing to lists of TIF image names, e.g.  `wcs_orinoquia_trial_region_201301_201512_-69.74_3.742.tif`.

The splits are over tiles. 

The image tiles and their label masks are stored together (not in separate folders for each split). Splitting is done by the datasets and loaders. 

### Chip size

Using 256 for the width and height of the chips. 


## Experiments

### `200211_mini_baseline`

Using bands that seem most helpful by looking at what each have been used for and the visualization of each band

Bands used:

2, 3, 4, 5, 6, 7

Bands 6 and 7 have fairly different pixel value distributions, so including both. 

Bands 4 and 5 are combined to give the NDVI, so the channels are

2, 3, 6, 7, NDVI  

(stacked in this order)

We are keeping them as float32, which they're read by rasterio as, to retain the original amount of information. We apply clipping, normalization and normalization according to the parameters specified in the experiment config file (values of max, min and gamma - in this experiment gamma is 1, so no gamma correction), all the while keeping them in float32 dtype (`return_array` set as `True` returns float32 instead of uint8). 


### Data augmentation techniques

- Undersample chips that only contain the most common classes


### Which bands are most helpful

Band 8 panchromatic needs to be downsampled; band 10 and 11 needs to be enlarged


## Flooding and seasonality

Water has very low values on NDVI plots since it does not reflect much NIR. So getting the greenest pixel should 
be okay for flooded savanna. 
 