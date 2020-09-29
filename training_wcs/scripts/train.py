# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Training script that trains a model instantiated in the configuration file specified.

Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import argparse
import importlib
import logging
import os
import shutil
import sys
import math
from datetime import datetime
from shutil import copytree
from time import localtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import Logger
from utils.train_utils import AverageMeter, log_sample_img_gt, render_prediction

# ai4eutils needs to be on the PYTHONPATH
from geospatial.enums import ExperimentConfigMode

# from azureml.core.run import Run  # importing this would make all the logging.info not output to stdout


logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='{asctime} {levelname} {message}',
                    style='{',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f'Using PyTorch version {torch.__version__}.')

parser = argparse.ArgumentParser(description='UNet training_wcs')
parser.add_argument(
    '--config_module_path',
    help="Path to the .py file containing the experiment's configurations")
args = parser.parse_args()

# config for the run
try:
    module_name = 'config'
    spec = importlib.util.spec_from_file_location(module_name, args.config_module_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config
    spec.loader.exec_module(config)
except Exception as e:
    logging.error(f'Failed to import the configurations. Exception: {e}')
    sys.exit(1)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}.')

torch.set_default_dtype(config.dtype)


def get_sample_images(which_set='train'):
    """
    Get a deterministic set of images in the specified set (train or val) by using the dataset and
    not the dataloader. Only works if the dataset is not IterableDataset.

    Args:
        which_set: one of 'train' or 'val'

    Returns:
        samples: a dict with keys 'chip' and 'chip_label', pointing to torch Tensors of
        dims (num_chips_to_visualize, channels, height, width) and (num_chips_to_visualize, height, width)
        respectively
    """
    assert which_set == 'train' or which_set == 'val'

    dataset = dset_train if which_set == 'train' else dset_val

    num_to_skip = 5  # first few chips might be mostly blank
    assert len(dataset) > num_to_skip + config.num_chips_to_viz

    keep_every = math.floor((len(dataset) - num_to_skip) / config.num_chips_to_viz)
    sample_idx = range(num_to_skip, len(dataset), keep_every)
    samples = dataset[sample_idx]
    return samples


def visualize_result_on_samples(model, sample_images, logger, step, split='train'):
    model.eval()
    with torch.no_grad():
        sample_images = sample_images.to(device=device)
        scores = model(sample_images)  # these are scores before the final softmax

        # TODO Juan suggests that we also visualize the second or third most confident classes predicted
        _, preds = scores.max(1)
        preds = preds.cpu().numpy()
        images_li = []
        for i in range(preds.shape[0]):
            preds_i = preds[i, :, :].squeeze()
            im, buf = render_prediction(preds_i)
            images_li.append((im, buf))

        logger.image_summary(split, 'result', images_li, step)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        logging.info('Initializing weights with Kaiming uniform')
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()


def train(loader, model, criterion, optimizer, epoch, step, logger_train):
    for batch_idx, data in enumerate(loader):
        # put model to training_wcs mode; we put it in eval mode in visualize_result_on_samples for every print_every
        model.train()
        step += 1

        x = data['chip'].to(device=device)  # move to device, e.g. GPU
        y = data['chip_label'].to(device=device)  # y is not a int value here; also an image

        # forward pass on this batch
        scores = model(x)
        loss = criterion(scores, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

        # TensorBoard logging and print a line to stdout; note that the accuracy is wrt the current mini-batch only
        if step % config.print_every == 1:
            _, preds = scores.max(1)  # equivalent to preds = scores.argmax(1)
            accuracy = (y == preds).float().mean()

            info = {'minibatch_loss': loss.item(), 'minibatch_accuracy': accuracy.item()}
            for tag, value in info.items():
                logger_train.scalar_summary(tag, value, step)

            logging.info(
                'Epoch {}, step {}, train minibatch_loss is {}, train minibatch_accuracy is {}'.format(
                    epoch, step, info['minibatch_loss'], info['minibatch_accuracy']))
    return step


def evaluate(loader, model, criterion):
    """Evaluate the model on dataset of the loader"""
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()  # put model to evaluation mode
    with torch.no_grad():
        for t, data in enumerate(loader):
            x = data['chip'].to(device=device)  # move to device, e.g. GPU
            y = data['chip_label'].to(device=device, dtype=torch.long)

            scores = model(x)

            loss = criterion(scores, y)
            # DEBUG logging.debug('Val loss = %.4f' % loss.item())

            _, preds = scores.max(1)
            accuracy = (y == preds).float().mean()

            losses.update(loss.item(), x.size(0))
            accuracies.update(accuracy.item(), 1)  # average already taken for accuracy for each pixel

    return losses.avg, accuracies.avg


def save_checkpoint(state, is_best, checkpoint_dir='../checkpoints'):
    """
    checkpoint_dir is used to save the best checkpoint if this checkpoint is best one so far
    """
    checkpoint_path = os.path.join(checkpoint_dir,
                                   f"checkpoint_epoch{state['epoch'] - 1}_"
                                   f"{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.pth.tar")
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def main():

    out_dir = './outputs' if config.out_dir is None else config.out_dir

    # if running locally, copy current version of scripts and config to output folder as a record
    if out_dir != './outputs':
        scripts_copy_dir = os.path.join(out_dir, 'repo_copy')
        cwd = os.getcwd()
        logging.info(f'cwd is {cwd}')
        if 'scripts' not in cwd:
            cwd = os.path.join(cwd, 'scripts')
        copytree(cwd, scripts_copy_dir)  # scripts_copy_dir cannot already exist
        logging.info(f'Copied over scripts to output dir at {scripts_copy_dir}')

    # create checkpoint dir
    checkpoint_dir = os.path.join(out_dir, config.experiment_name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # model
    model = config.model
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    if config.loss_weights is not None:
        assert isinstance(config.loss_weights, torch.Tensor), \
            'config.loss_weight needs to be of Tensor type'
        assert len(config.loss_weights) == config.num_classes, \
            f'config.loss_weight has length {len(config.loss_weights)} but needs to equal to num_classes'
    criterion = nn.CrossEntropyLoss(weight=config.loss_weights).to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=config.init_learning_rate)

    # resume from a checkpoint if provided
    starting_checkpoint_path = config.starting_checkpoint_path
    if starting_checkpoint_path and os.path.isfile(starting_checkpoint_path):
        logging.info('Loading checkpoint from {}'.format(starting_checkpoint_path))
        checkpoint = torch.load(starting_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

        # don't load the optimizer settings so that a newly specified lr can take effect
        # optimizer.load_state_dict(checkpoint['optimizer'])

        starting_epoch = checkpoint['epoch']  # we incremented epoch before saving it, so can just start here
        step = checkpoint.get('step', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        logging.info(f'Loaded checkpoint, starting epoch is {starting_epoch}, step is {step}, '
                     f'best accuracy is {best_acc}')
    else:
        logging.info('No valid checkpoint is provided. Start to train from scratch...')
        model.apply(weights_init)
        starting_epoch = 0
        best_acc = 0.0
        step = 0

    # data sets and loaders, which will be added to the global scope for easy access in other functions
    global dset_train, loader_train, dset_val, loader_val

    dset_train = config.dset_train
    loader_train = config.loader_train

    dset_val = config.dset_val
    loader_val = config.loader_val

    logging.info('Getting sample chips from val and train set...')
    samples_val = get_sample_images(which_set='val')
    samples_train = get_sample_images(which_set='train')

    # logging
    # run = Run.get_context()
    aml_run = None
    logger_train = Logger('train', config.log_dir, config.batch_size, aml_run)
    logger_val = Logger('val', config.log_dir, config.batch_size, aml_run)
    log_sample_img_gt(logger_train, logger_val,
                      samples_train, samples_val)
    logging.info('Logged image samples')

    if config.config_mode == ExperimentConfigMode.EVALUATION:
        val_loss, val_acc = evaluate(loader_val, model, criterion)
        logging.info(f'Evaluated on val set, loss is {val_loss}, accuracy is {val_acc}')
        return

    for epoch in range(starting_epoch, config.total_epochs):
        logging.info(f'\nEpoch {epoch} of {config.total_epochs}')

        # train for one epoch
        # we need the `step` concept for TensorBoard logging only
        train_start_time = datetime.now()
        step = train(loader_train, model, criterion, optimizer, epoch, step, logger_train)
        train_duration = datetime.now() - train_start_time

        # evaluate on val set
        logging.info('Evaluating model on the val set at the end of epoch {}...'.format(epoch))

        eval_start_time = datetime.now()
        val_loss, val_acc = evaluate(loader_val, model, criterion)
        eval_duration = datetime.now() - eval_start_time

        logging.info(f'\nEpoch {epoch}, step {step}, val loss is {val_loss}, val accuracy is {val_acc}\n')
        logger_val.scalar_summary('val_loss', val_loss, step)
        logger_val.scalar_summary('val_acc', val_acc, step)

        # visualize results on both train and val images
        visualize_result_on_samples(model, samples_train['chip'], logger_train, step, split='train')
        visualize_result_on_samples(model, samples_val['chip'], logger_val, step, split='val')

        # log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger_train.histo_summary(tag, value.data.cpu().numpy(), step)
            logger_train.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

        # record the best accuracy; save checkpoint for every epoch
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        logging.info(
            f'Iterated through {step * config.batch_size} examples. Saved checkpoint for epoch {epoch}. '
            f'Is it the highest accuracy checkpoint so far: {is_best}\n')

        save_checkpoint({
            # add 1 so when we restart from it, we can just read it and proceed
            'epoch': epoch + 1,
            'step': step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,
            'best_acc': best_acc
        }, is_best, checkpoint_dir)

        # log execution time for this epoch
        logging.info((f'epoch training_wcs duration is {train_duration.total_seconds()} seconds;'
                     f'evaluation duration is {eval_duration.total_seconds()} seconds'))
        logger_val.scalar_summary('epoch_duration_train', train_duration.total_seconds(), step)
        logger_val.scalar_summary('epoch_duration_val', eval_duration.total_seconds(), step)


if __name__ == '__main__':
    main()
