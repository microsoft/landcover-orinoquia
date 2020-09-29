# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
The Logger class logs metrics (scalar, image, and histograms) to an events file for display in TensorBoard.

Adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
Reference https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
"""

import os

import torch
import tensorflow as tf
import numpy as np


class Logger(object):

    def __init__(self, split, log_dir, batch_size, aml_run=None):
        """Create a summary writer logging to log_dir.
            Assumes that the batch size is constant throughout the run. We log
            step * batch_size to be consistent across runs of different batch size.
        """

        log_dir = os.path.join(log_dir, split)
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

        self.split = split
        self.batch_size = batch_size
        self.aml_run = aml_run

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""

        step = step * self.batch_size
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

        if self.aml_run:
            self.aml_run.log(f'{self.split}/{tag}', value)

    def image_summary(self, split, tag, images_and_buffers, step):
        """Log a list of images."""

        step = step * self.batch_size

        img_summaries = []
        for i, (pil_im, buf) in enumerate(images_and_buffers):
            # Create an Image object
            h = pil_im.shape[0] if isinstance(pil_im, torch.Tensor) else pil_im.size[0]
            w = pil_im.shape[1] if isinstance(pil_im, torch.Tensor) else pil_im.size[1]
            img_summary = tf.compat.v1.Summary.Image(encoded_image_string=buf.getvalue(),
                                       height=h,
                                       width=w)
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag=f'{split}_{i}/{tag}', image=img_summary))

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        step = step * self.batch_size

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
