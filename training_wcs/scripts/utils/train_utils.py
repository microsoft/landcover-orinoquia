# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import img_as_ubyte
from PIL import Image

from viz_utils import VizUtils


class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """

        Args:
            val: mini-batch loss or accuracy value
            n: mini-batch size
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


viz_util = VizUtils()
ndvi_normalizer = mcolors.Normalize(vmin=-1, vmax=1)

def log_sample_img_gt(logger_train, logger_val,
                      samples_train, samples_val):
    """
    Logs sample chips and ground truth labels to TensorBoard
    Args:
        logger_train: Logger object for the training_wcs loop, used every print_every
        logger_val: Logger object for the validation loop, used at the end of every epoch
        samples_train: dict of tensors of chips and labels for the training_wcs set
        samples_val: dict of tensors of chips and labels for the val set
    Returns:
        None
    """
    for split, samples, logger in [
        ('train', samples_train, logger_train),
        ('val', samples_val, logger_val)
    ]:
        chips = samples['chip']
        labels = samples['chip_label']

        print()
        print(f'{split} chips has shape {chips.shape}')

        tag = 'image bands 6, 3, 2'
        band_combo = (2, 1, 0)  # bands 6, 3, 2 in the baseline case
        images_and_buffers = []
        for i, sample in enumerate(chips):
            im_np = sample[band_combo, :, :]
            im_np = np.transpose(im_np, axes=(1, 2, 0))

            # if I don't print this it has min and max values outside of [-1, 1] ???
            # what race condition could this be???
            print(f'{i}th sample shape is {im_np.shape}, min: {im_np.min()}, max: {im_np.max()}')

            im = Image.fromarray(img_as_ubyte(im_np))
            buf = BytesIO()
            im.save(buf, format='png')
            images_and_buffers.append((im_np, buf))  # image_summary reads the first two dims for height and width

        logger.image_summary(split, tag, images_and_buffers, step=0)

        # log the NDVI of the sample chips
        tag = 'ndvi'
        ndvi_band = 4  # bands are (2, 3, 6, 7, NDVI, elevation) in baseline
        images_and_buffers = []
        for sample in chips:
            ndvi = sample[ndvi_band, :, :].squeeze()

            # TODO move to viz_utils as well
            fig = plt.figure(figsize=(4, 4))
            fig = plt.imshow(ndvi, cmap='viridis', norm=ndvi_normalizer)

            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            im = Image.open(buf)
            images_and_buffers.append((im, buf))
        logger.image_summary(split, tag, images_and_buffers, step=0)

        # log the elevation of the sample chips if available; elevation is available in the tensor input as a channel
        # note that here we do not set a normalizer for imshow() so it's stretched by the max and min values
        # on the chip only - not absolute compared to all elevation values
        if chips[0].shape[0] == 6:
            tag = 'elevation'
            images_and_buffers = []

            for sample in chips:
                elevation = sample[5, :, :].squeeze()

                im, buf = viz_util.show_single_band(elevation, size=(4, 4))

                images_and_buffers.append((im, buf))
            logger.image_summary(split, tag, images_and_buffers, step=0)

        # log the labels
        tag = 'label'
        images_and_buffers = []

        for label_mask in labels:
            im, buf = viz_util.show_label_raster(label_mask, size=(4, 4))
            images_and_buffers.append((im, buf))
        logger.image_summary(split, tag, images_and_buffers, 0)


def render_prediction(hardmax):
    im, buf = viz_util.show_label_raster(hardmax)
    return im, buf

