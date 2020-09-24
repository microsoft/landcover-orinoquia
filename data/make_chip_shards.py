# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
make_data_shards.py

This is an additional pre-processing step after tile_and_mask.py to cut chips out of the tiles
and store them in large numpy arrays, so they can all be loaded in memory during training.

The train and val splits will be stored separately to distinguish them.

This is an improvement on the original approach of chipping during training using LandsatDataset, but it is an
extra step, so each new experiment requiring a different input size/set of channels would need to re-run
this step. Data augmentation is still added on-the-fly.

Example invocation:
```
export AZUREML_DATAREFERENCE_wcsorinoquia=/boto_disk_0/wcs_data/tiles/full_sr_median_2013_2014

python data/make_chip_shards.py --config_module_path training_wcs/experiments/elevation/elevation_2_config.py --out_dir /boto_disk_0/wcs_data/shards/full_sr_median_2013_2014_elevation
```
"""

import argparse
import importlib
import os
import sys
import math
import numpy as np

from tqdm import tqdm


def create_shard(dataset, num_shards):
    """Iterate through the dataset to produce shards of chips as numpy arrays, for imagery input and labels.

    Args:
        dataset: an instance of LandsatDataset, which when iterated, each item contains fields
                    'chip' and 'chip_label'
        num_shards: number of numpy arrays to store all chips in

    Returns:
        returns a 2-tuple, where
        - the first item is a list of numpy arrays of dimension (num_chips, channel, height, width) with
          dtype float for the input imagery chips
        - the second item is a list of numpy arrays of dimension (num_chips, height, width) with
          dtype int for the label chips.
    """
    input_chips, label_chips = [], []
    for item in tqdm(dataset):
        # not using chip_id and chip_for_display fields
        input_chips.append(item['chip'])
        label_chips.append(item['chip_label'])

        # debugging
        # if len(input_chips) > 200:
        #     break
    num_chips = len(input_chips)
    print(f'Created {num_chips} chips.')

    items_per_shards = math.ceil(num_chips / num_shards)
    shard_idx = []
    for i in range(num_shards):
        shard_idx.append(
            (i * items_per_shards, (1 + i) * items_per_shards)
        )
   # print(f'Debug - shard_end_idx is {shard_idx}')

    print('Stacking imagery and label chips into shards')
    input_chip_shards, label_chip_shards = [], []
    for begin_idx, end_idx in shard_idx:
        if begin_idx < num_chips:
            input_chip_shard = input_chips[begin_idx:end_idx]
            input_chip_shard = np.stack(input_chip_shard, axis=0)
            print(f'dim of input chip shard is {input_chip_shard.shape}, dtype is {input_chip_shard.dtype}')
            input_chip_shards.append(input_chip_shard)

            label_chip_shard = label_chips[begin_idx:end_idx]
            label_chip_shard = np.stack(label_chip_shard, axis=0)
            print(f'dim of label chip shard is {label_chip_shard.shape}, dtype is {label_chip_shard.dtype}')
            label_chip_shards.append(label_chip_shard)

    return (input_chip_shards, label_chip_shards)


def save_shards(out_dir, set_name, input_chip_shards, label_chip_shards):
    for i_shard, (input_chip_shard, label_chip_shard) in enumerate(zip(input_chip_shards, label_chip_shards)):
        shard_path = os.path.join(out_dir, f'{set_name}_chips_{i_shard}.npy')
        np.save(shard_path, input_chip_shard)
        print(f'Saved {shard_path}')

        shard_path = os.path.join(out_dir, f'{set_name}_labels_{i_shard}.npy')
        np.save(shard_path, label_chip_shard)
        print(f'Saved {shard_path}')


def main():
    parser = argparse.ArgumentParser(description='Make data shards of chips from tiles')
    parser.add_argument(
        '--config_module_path',
        required=True,
        help=("Path to the experiment's configuration file, so that it can be imported as a module")
    )
    parser.add_argument(
        '--out_dir',
        help='Path to a dir where the numpy arrays are saved as pickle files'
    )
    parser.add_argument(
        '--train_shards',
        default=1,
        type=int,
        help='Number of numpy arrays to target for training set examples'
    )
    parser.add_argument(
        '--val_shards',
        default=1,
        type=int,
        help='Number of numpy arrays to target for validation set examples'
    )

    args = parser.parse_args()

    # config for the training run that generated the model checkpoint
    try:
        module_name = 'config'
        spec = importlib.util.spec_from_file_location(module_name, args.config_module_path)
        config = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = config
        spec.loader.exec_module(config)
    except Exception as e:
        print(f'Failed to import the configurations. Exception: {e}')
        sys.exit(1)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('Iterating through the training set to generate chips...')
    train_set = config.dset_train
    train_set.shuffle_tiles = True
    train_input_chip_shards, train_label_chip_shards = create_shard(train_set, args.train_shards)
    save_shards(out_dir, 'train', train_input_chip_shards, train_label_chip_shards)

    del train_input_chip_shards
    del train_label_chip_shards

    print('Iterating through the val set to generate chips...')
    val_set = config.dset_val
    val_set.shuffle_tiles = True
    val_input_chip_shards, val_label_chip_shards = create_shard(val_set, args.val_shards)
    save_shards(out_dir, 'val', val_input_chip_shards, val_label_chip_shards)

    print('Done!')


if __name__ == '__main__':
    main()
