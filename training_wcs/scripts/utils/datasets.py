# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This script contains two classes used for iterating chips in the dataset:
- LandsatDataset: reads one chip at a time from tiles in disk (so very slow)
- SingleShardChipsDataset:
"""

import logging
import math
import os
from random import shuffle

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data.dataset import IterableDataset, Dataset

LABELS_TO_USE = 'tiles_masks_coarse'  # 'tiles_masks'  - which set of labels to use


class LandsatDataset(IterableDataset):
    """Class representing a dataset of Landsat image chips and label masks, such as a training_wcs set.
        It reads one chip at a time from tiles stored in disk.
        make_chip_shards.py iterate through an instance of this class for train and val set and store the chips
        in numpy arrays for faster iteration during training_wcs.
    """

    def __init__(self, data_dir, tile_names, get_chip_func,
                 get_chip_mask_func=None, repeat_chip_func=None,
                 chip_size=256, tile_size=2000, transform=None):
        """
        Args:
            data_dir (string): Directory containing the subfolders tiles and tiles_label_masks
            tile_names (list of strings): Names of tiles in the split corresponding to this Dataset
            get_chip_func (function): A function that returns (chip, chip_for_display) given the arguments
                (rasterio reader, chip_window)
            get_chip_mask_func (function): A function that returns chip_mask given the arguments
                (rasterio reader, chip_window for PIL image)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LandsatDataset).__init__()

        assert len(tile_names) > 0, 'tile_names passed to LandsatDataset is empty'

        self.data_dir = data_dir
        self.get_chip_func = get_chip_func
        self.get_chip_mask_func = get_chip_mask_func
        self.repeat_chip_func = repeat_chip_func
        self.chip_size = chip_size
        self.tile_names = tile_names
        self.num_tiles = len(tile_names)

        # how many rows and columns of chips can fit on this tile
        # pixels per side / chip_size of 256
        self.num_rows = math.ceil(tile_size / chip_size)
        self.num_cols = self.num_rows  # only support square tiles and chips
        print(f'Number of rows or columns in a tile for making chips is {self.num_rows}')

        self.transform = transform

        self._shuffle_tiles = False

    @property
    def shuffle_tiles(self):
        return self._shuffle_tiles

    @shuffle_tiles.setter
    def shuffle_tiles(self, value):
        assert isinstance(value, bool)
        self._shuffle_tiles = value

    def __iter__(self):
        # shuffle the tiles if the shuffle_tiles option is on (each epoch will get a new iterator instance, I think)
        if self._shuffle_tiles:
            print('Returning iterator instance with the tiles shuffled!')
            shuffle(self.tile_names)
        else:
            print('Returning iterator instance with the tiles NOT shuffled!')
        return self._get_generator()

    @staticmethod
    def _default_get_chip_mask(tile_label, chip_window):
        """

        Args:
            tile_label: PIL Image of the tile's label mask
            chip_window: PIL window (left x, upper y, right x2, lower y2)

        Returns:
            chip_label as a numpy array of dtype np.uint8
        """
        chip_label = tile_label.crop(box=chip_window)
        chip_label = np.array(chip_label, dtype=np.uint8)
        return chip_label


    def _get_chip(self, tile_reader, tile_label, col_idx, row_idx):
        """Call the config supplied functions to get the imagery chip and optionally the label mask chip
        referencing the col_idx and row_idx specified.

        This function is responsible for
        - observing entirely empty chips and masking out imagery pixels corresponding to label masks where
          there is no label (category of 0).
        - masking out label mask where there is no valid imagery available
        (i.e. both imagery and label chips should be empty / of value 0 in the same pixels)

        Args:
            tile_reader: rasterio dataset object of the imagery tile
            tile_label: PIL Image of the tile's label mask
            col_idx: index of the column in the tile this call should reference
            row_idx: index of the row in the tile this call should reference

        Returns:
            (chip, chip_label_masked, chip_for_display)
        """

        # torch dimensions: batch, channel, height, width

        # rasterio window is (col_off x, row_off y, width, height)
        chip_window = (
            col_idx * self.chip_size,
            row_idx * self.chip_size,
            self.chip_size,
            self.chip_size
        )
        chip, chip_for_display = self.get_chip_func(tile_reader, chip_window)  # dim is (C, H, W)

        # where cloud is masked out on GEE, there is NaN values
        # pixel values are normalized to [0, 1]; NDVI is [-1, 1]
        chip = np.nan_to_num(chip, nan=0.0, posinf=1.0, neginf=-1.0)

        # return None if satellite imagery in this chip is empty / not available
        # sometimes the SWIR has non-zero values while all other bands are zero, so
        # using band 2 (visible blue), which is the first in the stack, to mask out labels
        if chip[0].max() == chip[0].min():
            logging.debug('_get_chip, Empty imagery chip!')
            return None, None, None

        # mask for where data is available in the satellite chip,
        # again using band 2 (visible blue) to determine if the satellite chip is available
        # as opposed to summing across all bands
        # also avoids the problem that the elevation layer has no gaps
        sat_mask = chip[0].squeeze() > 0.0

        # PIL window is (left x, upper y, right x2, lower y2)
        mask_chip_window = (
            col_idx * self.chip_size,
            row_idx * self.chip_size,
            col_idx * self.chip_size + self.chip_size,
            row_idx * self.chip_size + self.chip_size
        )
        if self.get_chip_mask_func is None:
            chip_label = LandsatDataset._default_get_chip_mask(tile_label, chip_window=mask_chip_window)
        else:
            chip_label = self.get_chip_mask_func(tile_label, chip_window=mask_chip_window)

        if chip_label is None:
            print('_get_chip, Skipped due to label mask not meeting criteria')

        # the label mask can also be empty on some places where we have imagery data
        if chip_label.max() == 0 and chip_label.min() == 0:
            logging.debug('_get_chip, Empty label mask chip!')
            return None, None, None

        label_available_mask = chip_label > 0

        # we only want to keep pixels that have both data and label
        data_label_mask = np.logical_and(sat_mask, label_available_mask)

        chip = chip * data_label_mask
        chip = chip * sat_mask  # masking out the elevation and other bands that may not be zero where the blue band is

        if chip_for_display.shape != (self.chip_size, self.chip_size, 3):
            # default pad constant value is 0
            chip_for_display = np.pad(chip_for_display,
                                      [(0, self.chip_size - chip_for_display.shape[0]),
                                       (0, self.chip_size - chip_for_display.shape[1]),
                                       (0, 0)])
        chip_for_display = chip_for_display * np.expand_dims(data_label_mask, axis=2)  # can't broadcast here somehow

        chip_label_masked = chip_label * data_label_mask

        # these are without the batch dimension, which will be added by the default collate fn as the outmost/first dim
        return chip, chip_label_masked, chip_for_display

    def _get_generator(self):
        num_chips = 0
        for tile_idx, tile_name in enumerate(self.tile_names):
            # print(f'Now accessing tile number {tile_idx}, {tile_name}')
            tile_path = os.path.join(self.data_dir, 'tiles', tile_name)

            parts = tile_name.split('.tif')[0].split('_')
            lon = parts[-2]
            lat = parts[-1]

            mask_name = f'mask_{lon}_{lat}.tif'
            mask_path = os.path.join(self.data_dir, LABELS_TO_USE, mask_name)

            tile_reader = rasterio.open(tile_path)
            tile_label = Image.open(mask_path)

            for row_idx in range(self.num_rows):
                for col_idx in range(self.num_cols):
                    chip, chip_label, chip_for_display = self._get_chip(tile_reader, tile_label, col_idx, row_idx)

                    if chip is None:
                        continue  # completely empty chip

                    sample = {
                        'chip_id': f'{tile_name}_col{col_idx}_row{row_idx}',
                        'chip': chip,
                        'chip_label': chip_label,
                        'chip_for_display': chip_for_display
                    }
                    if self.transform:
                        sample = self.transform(sample)
                    num_chips += 1
                    yield sample

                    if self.repeat_chip_func is not None and self.repeat_chip_func(chip_label):
                        #print('Chip repeated because of repeat_chip_func!')
                        yield sample  # execution should return to this condition
                    else:
                        #print('Chip NOT repeated')
                        pass
        print(f'End of generator... Number of chips is {num_chips}')


class SingleShardChipsDataset(Dataset):
    """
    A class for iterating over chips created using the make_chip_shards.py script. The entire shard is loaded
    into memory as a numpy array, so iteration is fast.

    Only supports a single shard currently. To support multiple shards effectively we probably need to
    implement IterableDataset.
    """

    def __init__(self, data_shard_dir, shard_prefix='train', channels=None, transform=None):

        # available channels in this round are (2, 3, 6, 7, NDVI, elevation)
        # so if you want all the channels to be used, set channels=(0, 1, 2, 3, 4, 5)

        super(SingleShardChipsDataset).__init__()

        self.transform = transform

        self.chips = np.load(os.path.join(data_shard_dir, f'{shard_prefix}_chips_0.npy'))
        if channels is not None:
            assert max(channels) < self.chips.shape[1]
            assert min(channels) >= 0
            self.chips = self.chips[:, channels, :, :]

        self.labels = np.load(os.path.join(data_shard_dir, f'{shard_prefix}_labels_0.npy'))
        print(f'For prefix {shard_prefix}, loaded chips of dims {self.chips.shape}, labels of dims {self.labels.shape}')

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):

        sample = {
            'chip': self.chips[idx, :, :, :],
            'chip_label': self.labels[idx, :, :]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
