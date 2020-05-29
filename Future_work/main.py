from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import argparse
import torch
import numpy as np
from numpy.random import choice
import os
import cv2
import json
from scipy import ndimage
from scipy.misc import imread, imresize
import re

import sys
sys.path.insert(0, './MattNet/tools')
from mattnet import MattNet


class Parameters:
    def __init__(self):
        self.file_path = '/home/luoyunpeng/project/parser/dataset_coco_with_nps.json'
        self.image_root = '/home/luoyunpeng/dataset/COCO/'


def blur(sal_map, sigma=19):
    sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
    return sal_map


def check_or_create(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_att_maps(img_path, sentences, **kwargs):
    img_data = mattnet.forward_image(img_path, nms_thresh=0.3, conf_thresh=0.50)

    threshold = kwargs.get('threshold', 200)
    kernel_size = kwargs.get('kernel_size', 41)

    att_maps = []
    for sentence in sentences:
        attns = []
        boxes = []
        for expr in sentence['nps']:
            entry = mattnet.comprehend(img_data, expr)
            attns.append(entry['sub_grid_attn'])
            boxes.append(entry['pred_box'])

        img = imread(img_path)
        if len(img.shape) == 3:
            H, W, _ = img.shape
        else:
            H, W = img.shape
        att_map = combine_attns(attns, boxes, H, W, threshold)

        res = cv2.GaussianBlur(att_map, (kernel_size, kernel_size), 0)
        if np.max(res) - np.min(res) == 0:
            pass
        else:
            res = (res - np.min(res)) / (np.max(res) - np.min(res))

        att_maps.append(res)

    return att_maps


def get_mask(img_path, sentences):
    img_data = mattnet.forward_image(img_path, nms_thresh=0.3, conf_thresh=0.50)
    masks = []
    for sentence in sentences:
        attns = []
        boxes = []
        for expr in sentence['nps']:
            entry = mattnet.comprehend(img_data, expr)
            attns.append(entry['sub_grid_attn'])
            boxes.append(entry['pred_box'])

        img = imread(img_path)
        if len(img.shape) == 3:
            H, W, _ = img.shape
        else:
            H, W = img.shape
        mask = combine_boxes(boxes, H, W)
        masks.append(mask)
    return masks


def combine_attns(attns, boxes, H, W, threshold=None):
    res = np.zeros((H, W))
    mean_rec = []
    for attn, box in zip(attns, boxes):
        attn = np.array(attn).reshape(7, 7)
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        attn = imresize(attn, [h, w])
        mean_rec.append(np.mean(attn))
        res[y:y + h, x:x + w] = np.maximum(res[y:y + h, x:x + w], attn)
    res_mean = np.mean(mean_rec)
    res[res < res_mean] = 0
    return res


def combine_boxes(boxes, H, W):
    res = np.zeros((H, W))
    for box in boxes:
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        res[y:y + h, x:x + w] = 1
    return res


def sample(att_map):
    dis = att_map / np.sum(att_map)
    positions = np.arange(att_map.shape[0] * att_map.shape[1])

    draw = choice(positions, 100,
                  p=dis.reshape(-1), replace=False)
    assert len(set(draw)) == 100
    res = np.zeros_like(dis)
    for index in draw:
        res[np.unravel_index(index, res.shape)] = 1
    return res


def fixation2map(fixaitons):
    blurred_fixation_map = blur(fixaitons, sigma=31)

    blurred_fixation_map -= np.min(blurred_fixation_map)
    blurred_fixation_map /= np.max(blurred_fixation_map)

    blurred_fixation_map = (blurred_fixation_map * 255).astype('uint8')
    return blurred_fixation_map


if __name__ == '__main__':
    # test()

    # default arguments
    image_name_index = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco',
                        help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    parser.add_argument('--model_id', type=str, default='mrcn_cmr_with_st', help='model id name')
    parser.add_argument('--coco_caption_dataset', type=str, default='/home/luoyunpeng/')

    # data arguments
    parser.add_argument('--input', type=str, default='/Shared_Resources/social_media/DataSet/COCO/MM/image_name/' + str(image_name_index) + '.json')
    # parser.add_argument('--input', type=str, default='/Shared_Resources/social_media/DataSet/COCO/annotations/captions_train2014.json')

    parser.add_argument('--output_path', type=str, default='/Shared_Resources/social_media/DataSet/COCO/MM')
    # parser.add_argument('--output_path', type=str, default='/home/zhangxuying/Project/data/pseudo_mattnet_coco_train')
    parser.add_argument('--max_pool', action='store_true', default=False)

    args = parser.parse_args()

    file_path = '/Shared_Resources/luo/project/parser/dataset_coco_with_nps.json'
    image_root = '/Shared_Resources/social_media/DataSet/COCO/train2014'

    with open(file_path) as f:
        infos = json.load(f)
    print(infos.keys())
    images = infos['images']
    print(images[0]['cocoid'])
    cocoid_keyed_images = {image['cocoid']: image for image in images}

    # coco_anno_path = args.input
    # with open(coco_anno_path) as f:
    #     coco_infos = json.load(f)
    # print(coco_infos.keys())
    # coco_image_infos = coco_infos['images']
    # images = [re.findall(r'(.+?)\.', image['file_name'])[0] for image in coco_image_infos]
    # print('image num:', len(images))

    with open(args.input, 'r') as f:
        image_names = json.load(f)
    images = [re.findall(r'(.+?)\.', image)[0] for image in image_names]
    print('image num:', len(images))

    print('start')
    mattnet = MattNet(args)

    for index, image in enumerate(images):
        # special_images = [
        #     'COCO_train2014_000000004308'
        # ]
        # if image in special_images:
        #     continue
        print("processing {}.jpg".format(image))
        image_path = os.path.join(image_root, image + '.jpg')
        cocoid = int(image[-6:])
        sentences = cocoid_keyed_images[cocoid]['sentences']
        with torch.no_grad():
            att_maps = get_att_maps(image_path, sentences)

            res = sum(att_maps) / 5
            fixation = sample(res)
            fixation_map = fixation2map(fixation)

            save_fixation_path = os.path.join(args.output_path, 'fixation', str(image_name_index), image + '.npy')
            np.save(save_fixation_path, fixation)
            save_fixation_map_path = os.path.join(args.output_path, 'map', str(image_name_index), image + '.png')
            cv2.imwrite(save_fixation_map_path, fixation_map)

            if index % 10 == 0:
                print('{}/{}'.format(index, len(images)))

        torch.cuda.empty_cache()

    print('finished')
