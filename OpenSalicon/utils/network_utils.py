# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import math
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime as dt


class MultiLableNLLWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MultiLableNLLWithLogitsLoss, self).__init__()
        self.softmax2d = nn.Softmax2d()

    def forward(self, output, target):
        assert output.shape == target.shape
        bs, C, H, W = output.shape
        output = output.view(bs, -1)
        log_softmax = F.log_softmax(output, dim=1).view(bs, C, H, W)
        loss = - log_softmax * target
        return torch.sum(loss) / torch.sum(target)


class MaskLoglikelihoodLoss(nn.Module):
    def __init__(self):
        super(MaskLoglikelihoodLoss, self).__init__()

    def forward(self, output, target):
        assert output.shape == target.shape
        # print('output')
        # print(output)
        # print('target')
        # print(target)
        loss = - output * target
        return torch.sum(loss) / torch.sum(target)


def normalize_map(output):
    # print(output)
    max_each_sample, _ = output.view(output.size(0), -1).max(1)
    min_each_sample, _ = output.view(output.size(0), -1).min(1)
    assert max_each_sample.shape[0] == output.size(0)
    max_each_sample = max_each_sample.view(-1, 1, 1, 1)
    min_each_sample = min_each_sample.view(-1, 1, 1, 1)

    # print(max_each_sample)
    # print(min_each_sample)
    return (output - min_each_sample) / (max_each_sample - min_each_sample)


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
            type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, model, model_solver, best_nss, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_nss': best_nss,
        'best_epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'model_solver_state_dict': model_solver.state_dict()
    }
    torch.save(checkpoint, file_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_gaussian_kernel(kernel_size=5, sigma=3, n_channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(n_channels, 1, 1, 1)

    return gaussian_kernel


def get_center_surrounded_filter(filter_size, center_size, n_channels=1):
    n_padding = (filter_size - center_size) // 2
    pos_value = 1 / center_size ** 2
    neg_value = -1 / (filter_size ** 2 - center_size ** 2)

    csf = np.full((center_size, center_size), pos_value).astype(np.float32)
    csf = np.pad(csf, n_padding, 'constant', constant_values=neg_value)
    csf = torch.from_numpy(csf).view(1, 1, filter_size, filter_size)
    csf = csf.repeat(1, n_channels, 1, 1)

    return var_or_cuda(csf)


def get_center_bias(img_shape, center_bias_path):
    CENTER_BIAS_SIZE = 1024
    center_bias = np.load(center_bias_path).astype(np.float32)
    center_bias = scipy.ndimage.zoom(
        center_bias, (img_shape[0] / CENTER_BIAS_SIZE, img_shape[1] / CENTER_BIAS_SIZE),
        order=0, mode='nearest')
    center_bias = center_bias[np.newaxis, np.newaxis, :, :]
    center_bias = torch.from_numpy(center_bias)
    # center_bias = torch.exp(center_bias)
    # center_bias = center_bias / torch.sum(center_bias)
    # print('torch.sum(center_bias) = ', torch.sum(center_bias))
    return var_or_cuda(center_bias)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
