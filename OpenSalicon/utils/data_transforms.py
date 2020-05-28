# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import numpy as np
import torch
from random import random
import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.config import cfg


class Compose(object):
    """ Composes several transforms together.
    For example:
    # >>> transforms.Compose([
    # >>>     transforms.RandomBackground(),
    # >>>     transforms.CenterCrop(127, 127, 3),
    # >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, fixation_map):
        for t in self.transforms:
            image, fixation_map = t(image, fixation_map)

        return image, fixation_map


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, image, fixation_map):
        assert (isinstance(image, np.ndarray))
        assert (isinstance(fixation_map, np.ndarray))
        image = np.transpose(image, (2, 0, 1))
        img_tensor = torch.from_numpy(image)
        fixation_map = np.transpose(fixation_map, (2, 0, 1))
        sm_tensor = torch.from_numpy(fixation_map)

        return img_tensor, sm_tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, fixation_map):
        assert (isinstance(image, np.ndarray))
        image -= self.mean
        image /= self.std

        return image, fixation_map


class RandomPermuteRGB(object):
    def __call__(self, image, fixation_map):
        assert (isinstance(image, np.ndarray))
        random_permutation = np.random.permutation(3)
        image = image[:, :, random_permutation]

        return image, fixation_map


class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.crop_size_c = crop_size[2]

    def __call__(self, image, fixation_map):
        img_height, img_width, _ = image.shape

        if img_height > self.crop_size_h and img_width > self.crop_size_w:
            x_left = (img_width - self.crop_size_w) * 0.5
            x_right = x_left + self.crop_size_w
            y_top = (img_height - self.crop_size_h) * 0.5
            y_bottom = y_top + self.crop_size_h
        else:
            x_left = 0
            x_right = img_width
            y_top = 0
            y_bottom = img_height

        processed_image = cv2.resize(
            image[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        processed_map = cv2.resize(
            fixation_map[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        processed_map = processed_map[:, :, np.newaxis]
        # processed_map = processed_map.astype(np.bool).astype(np.float32)    # Normalize to binary maps
        return processed_image, processed_map


class CenterCrop2(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.crop_size_c = crop_size[2]

    def __call__(self, image, fixation_map):

        image = cv2.resize(image, (self.img_size_w, self.img_size_h))
        img_height, img_width, _ = image.shape

        if img_height > self.crop_size_h and img_width > self.crop_size_w:
            x_left = (img_width - self.crop_size_w) * 0.5
            x_right = x_left + self.crop_size_w
            y_top = (img_height - self.crop_size_h) * 0.5
            y_bottom = y_top + self.crop_size_h
        else:
            x_left = 0
            x_right = img_width
            y_top = 0
            y_bottom = img_height

        # processed_image = image[int(y_top):int(y_bottom), int(x_left):int(x_right)]
        # processed_map = fixation_map[int(y_top):int(y_bottom), int(x_left):int(x_right)]
        processed_image = cv2.resize(
            image[int(y_top):int(y_bottom), int(x_left):int(x_right)], (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W))
        processed_map = cv2.resize(
            fixation_map[int(y_top):int(y_bottom), int(x_left):int(x_right)], (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W))

        processed_map = processed_map[:, :, np.newaxis]

        # processed_map = processed_map.astype(np.bool).astype(np.float32)    # Normalize to binary maps
        return processed_image, processed_map


class RandomCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.crop_size_c = crop_size[2]

    def __call__(self, image, fixation_map):
        img_height, img_width, _ = image.shape

        if img_height > self.crop_size_h and img_width > self.crop_size_w:
            x_left = (img_width - self.crop_size_w) * random()
            x_right = x_left + self.crop_size_w
            y_top = (img_height - self.crop_size_h) * random()
            y_bottom = y_top + self.crop_size_h
        else:
            x_left = 0
            x_right = img_width
            y_top = 0
            y_bottom = img_height

        processed_image = cv2.resize(
            image[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        processed_map = cv2.resize(
            fixation_map[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        processed_map = processed_map[:, :, np.newaxis]
        # processed_map = processed_map.astype(np.bool).astype(np.float32)    # Normalize to binary maps
        return processed_image, processed_map


class RandomCrop2(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.crop_size_c = crop_size[2]

    def __call__(self, image, fixation_map):

        image = cv2.resize(image, (self.img_size_w, self.img_size_h))
        img_height, img_width, _ = image.shape

        if img_height > self.crop_size_h and img_width > self.crop_size_w:
            x_left = (img_width - self.crop_size_w) * random()
            x_right = x_left + self.crop_size_w
            y_top = (img_height - self.crop_size_h) * random()
            y_bottom = y_top + self.crop_size_h
        else:
            x_left = 0
            x_right = img_width
            y_top = 0
            y_bottom = img_height

        # processed_image = image[int(y_top):int(y_bottom), int(x_left):int(x_right)]
        # processed_map = fixation_map[int(y_top):int(y_bottom), int(x_left):int(x_right)]
        processed_image = cv2.resize(
            image[int(y_top):int(y_bottom), int(x_left):int(x_right)], (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W))
        processed_map = cv2.resize(
            fixation_map[int(y_top):int(y_bottom), int(x_left):int(x_right)],
            (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W))

        processed_map = processed_map[:, :, np.newaxis]

        # processed_map = processed_map.astype(np.bool).astype(np.float32)    # Normalize to binary maps
        return processed_image, processed_map


class RandomFlip(object):
    def __call__(self, image, fixation_map):
        if random() > 0.5:
            image = np.fliplr(image)
            fixation_map = np.fliplr(fixation_map)

        return image, fixation_map


class RandomGaussianNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, fixation_map):
        assert (isinstance(image, np.ndarray))
        height, width, channels = image.shape
        gaussian_noise = np.random.normal(self.mean, self.std, (height, width, channels))
        image += gaussian_noise

        return image, fixation_map
