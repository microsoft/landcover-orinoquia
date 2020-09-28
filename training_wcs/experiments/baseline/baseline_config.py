"""
Configurations for the 20200505_mini_baseline experiment
"""

import json
import os

import torch
import numpy as np

from viz_utils import VizUtils
from training_wcs.scripts.models.unet.unet import Unet  # models.unet.unet import Unet


experiment_name = 'wcs_baseline_202005'

eval_mode = True

# I/O -------------------------------------------------------------------------------------------------
if not eval_mode:
    aml_data_ref = os.environ.get('AZUREML_DATAREFERENCE_wcsorinoquia', '')
    assert len(aml_data_ref) > 0, 'Reading aml_data_ref from environment vars resulted in empty string.'

    data_dir = os.path.join(aml_data_ref, 'tiles', 'full_sr_median_2013_2014')
    assert 'tiles' in os.listdir(data_dir)
    assert 'tiles_masks' in os.listdir(data_dir)

    # a dir with experiment_name will be created in here and checkpoints are saved here
    # set as './outputs' for AML to stream to this Run's folder
    out_dir = '/boto_disk_0/wcs/20190518_feature_scale_1/outputs' # './outputs'
    os.makedirs(out_dir, exist_ok=True)

    # TF events go here. Set it as './logs' if using AML so they can be streamed
    log_dir = '/boto_disk_0/wcs/20190518_feature_scale_1/logs' # './logs'

    # train/val splits are stored in
    # on AML, this needs to be relative to the source_directory level
    splits_file = './constants/splits/full_sr_median_2013_2014_splits.json' # '../training_wcs/scripts/constants/splits/full_sr_median_2013_2014_splits.json'

    with open(splits_file) as f:
        splits = json.load(f)
    train_split = splits['train']
    val_split = splits['val']
    print(f'Train set has {len(train_split)} tiles; val set has {len(val_split)} tiles.')


# Training ----------------------------------------------------------------------------------------------

evaluate_only = False  # Only evaluate the model on the val set once

# this is the *total* epoch; if restarting from a checkpoint, be sure to add the additional number of epochs
# to fine-tune on top of the original value of this var
total_epochs = 1000
print_every = 100  # print every how many steps; just the minibatch loss and accuracy
assert print_every >= 1, 'print_every needs to be greater than or equal 1'

starting_checkpoint_path = None

init_learning_rate = 1e-4

batch_size = 24

# probability a chip is kept in the sample while sampling train and val chips at the start of training_wcs
# this should be smaller if we now have more training_wcs examples
# prob_keep_chip = 0.006
# 133 training_wcs tiles * 64 chips per tile = 8512 chips. Should keep every 177 if visualizing 48, which is 0.0056
keep_every = 30  # a balance between not covering all training_wcs tiles vs iterating through val tiles too many times

num_chips_to_viz = 48


# Hardware and framework --------------------------------------------------------------------------------
dtype = torch.float32


# Model -------------------------------------------------------------------------------------------------

num_classes = 34   # empty plus the 33 WCS classes; this is the number of output nodes

num_in_channels = 5  # 2, 3, 6, 7, NDVI

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
    12
]
less_common_classes = [
    32, 33
]

weights = []
for i in range(num_classes):
    if i in common_classes:
        weights.append(1)
    elif i in less_common_classes:
        weights.append(2)
    else:
        weights.append(10)
loss_weights = torch.FloatTensor(weights)  # None if no weighting for classes
print('Weights on loss per class used:')
print(loss_weights)

# how many subprocesses to use for data loading
# None for now - need to modify datasets.py to use
data_loader_num_workers = None

# not available in IterableDataset data_loader_shuffle = True  # True to have the data reshuffled at every epoch

chip_size = 256


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

viz_util = VizUtils()

def get_chip(tile_reader, chip_window):
    """

    Returns:
        A numpy array of dims (5, H, W)
    """
    normal_bands = [2, 3, 6, 7]  # bands to be used without calculating other indices e.g. NDVI
    bands_to_stack = []
    for b in normal_bands:
        # getting one band at a time because min, max and gamma may be different
        band = viz_util.show_landsat8_tile(
            tile_reader,
            bands=[b],  # pass in a list to get the batch dimension in the results
            window=chip_window,
            band_min=bands_normalization_params['min'][b],
            band_max=bands_normalization_params['max'][b],
            gamma=bands_normalization_params['gamma'][b],
            return_array=True
        )
        bands_to_stack.append(band)  # band is 2D (h, w), already squeezed, dtype is float 32

    ndvi = viz_util.show_landsat8_ndvi(tile_reader, window=chip_window)  # 2D, dtype is float32
    bands_to_stack.append(ndvi)

    stacked = np.stack(bands_to_stack)

    if stacked.shape != (5, chip_size, chip_size):
        # default pad constant value is 0
        stacked = np.pad(stacked,
                        [(0, 0), (0, chip_size - stacked.shape[1]), (0, chip_size - stacked.shape[2])])

    assert stacked.shape == (5, chip_size, chip_size), f'Landsat chip has wrong shape: {stacked.shape}, should be (5, h, w)'

    # prepare the chip for display
    chip_for_display = viz_util.show_landsat8_tile(tile_reader,
                                                   window=chip_window,
                                                   band_max=3000,  # what looks good for RGB
                                                   gamma=0.5,
                                                   return_array=True)
    return stacked, chip_for_display

