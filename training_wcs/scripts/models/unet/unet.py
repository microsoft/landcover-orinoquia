import logging

import torch.nn as nn

from .unet_utils import *

"""
Unet model definition. 

Code mostly taken from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py
"""


class Unet(nn.Module):

    def __init__(self, feature_scale=1,
                 n_classes=3, in_channels=3,
                 is_deconv=True, is_batchnorm=False):
        """A U-Net implementation.

        Args:
            feature_scale: the smallest number of filters (depth c) is 64 when feature_scale is 1,
                           and it is 32 when feature_scale is 2
            n_classes: number of output classes
            in_channels: number of channels in input
            is_deconv:
            is_batchnorm:
        """
        super(Unet, self).__init__()

        self._num_classes = n_classes

        assert 64 % feature_scale == 0, f'feature_scale {feature_scale} does not work with this UNet'

        filters = [64, 128, 256, 512, 1024]  # this is `c` in the diagram, [c, 2c, 4c, 8c, 16c]
        filters = [int(x / feature_scale) for x in filters]
        logging.info('filters used are: {}'.format(filters))

        # downsampling
        self.conv1 = UnetConv2(in_channels, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2(filters[3], filters[4], is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp(filters[4], filters[3], is_deconv)
        self.up_concat3 = UnetUp(filters[3], filters[2], is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], self._num_classes, kernel_size=1)
        self._filters = filters  # we need this info for re-training

    def forward(self, inputs, return_features=False):
        """If return_features is True, returns tuple (final outputs, last feature map),
        else returns final outputs only.
        """
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        if return_features:
            return final, up1
        else:
            return final

    def change_num_classes(self, new_num_classes: int):
        """Re-initialize the final layer with another number of output classes if different from
        existing number of classes
        """
        if new_num_classes == self._num_classes:
            return

        assert new_num_classes > 1, 'Number of classes need to be > 1'
        self._num_classes = new_num_classes

        self.final = nn.Conv2d(self._filters[0], self._num_classes, kernel_size=1)
        nn.init.kaiming_uniform_(self.final.weight)
        self.final.bias.data.zero_()
