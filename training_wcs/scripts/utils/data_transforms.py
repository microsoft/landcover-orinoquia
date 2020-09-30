# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Transforms to be performed on the loaded chip: ToTensor, RandomHorizontalFlip, and RandomVerticalFlip.

Referencing: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

We could use https://github.com/mlagunas/pytorch-nptransforms/ instead of using torch.flip (Caleb suggested)
"""
import random

import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {

            # 'chip_id': sample['chip_id'],
            # some tiles become float64 doubles for some reason if just using from_numpy (?)
            'chip': torch.from_numpy(sample['chip']).to(dtype=torch.float32),
            # torch.long is required to compute nn.CrossEntropyLoss
            'chip_label': torch.from_numpy(sample['chip_label']).to(dtype=torch.long),
            # 'chip_for_display': torch.from_numpy(sample['chip_for_display'])
        }


class RandomHorizontalFlip(object):
    """ Horizontally flip chip with probability 0.5

    Horizontal flip - flip along width dim

    Can use https://pytorch.org/docs/master/generated/torch.fliplr.html instead when it's available in > 1.5.1

    chip is of dim (channel, height, width)
    chip_for_display is of dim (height, width, channel)
    chip_label is of dim (height, width)
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            return {
                # 'chip_id': sample['chip_id'],
                'chip': torch.flip(sample['chip'], [2]),
                'chip_label': torch.flip(sample['chip_label'], [1]),
                # 'chip_for_display': torch.flip(sample['chip_for_display'], [1])
            }
        else:
            return sample


class RandomVerticalFlip(object):
    """ Vertically flip chip with probability 0.5

    Vertical flip - flip along height dim

    Can use https://pytorch.org/docs/master/generated/torch.flipud.html instead when it's available in > 1.5.1

    chip is of dim (channel, height, width)
    chip_for_display is of dim (height, width, channel)
    chip_label is of dim (height, width)
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            return {
                # 'chip_id': sample['chip_id'],
                'chip': torch.flip(sample['chip'], [1]),
                'chip_label': torch.flip(sample['chip_label'], [0]),
                # 'chip_for_display': torch.flip(sample['chip_for_display'], [0])
            }
        else:
            return sample
