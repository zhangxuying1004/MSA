from torch.utils.data import Dataset, DataLoader

import cv2
import json
import numpy as np
import os
from enum import Enum, unique
from datetime import datetime

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.config import cfg, params
from OpenSalicon.utils import data_transforms


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


@unique
class DatasetMode(Enum):
    Consistent = 0
    Diverse = 1


# salicon数据集的数据类别划分
DATASET_TYPE_MAPPING = {DatasetType.TRAIN: 'train', DatasetType.TEST: 'test', DatasetType.VAL: 'val'}
# salicon数据集的模态类别划分
DATASET_MODE_MAPPING = {DatasetMode.Diverse: 'diverse', DatasetMode.Consistent: 'consistent'}

# 设置 data augmentation
IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C
CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W, cfg.CONST.CROP_IMG_C
train_transforms = data_transforms.Compose([
    data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    data_transforms.RandomGaussianNoise(cfg.TRAIN.GAUSSIAN_NOISE_MEAN, cfg.TRAIN.GAUSSIAN_NOISE_STD),
    data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
    data_transforms.RandomPermuteRGB(),
    data_transforms.ToTensor(),
])
val_transforms = data_transforms.Compose([
    data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    data_transforms.ToTensor(),
])


class SaliconDataset(Dataset):
    def __init__(self, cfg, dataset_type, transforms=None):
        # Setup transforms
        self.transforms = transforms

        # Load all files of the dataset
        with open(cfg.DATASETS.SALICON.DATASET_FILE_PATH, encoding='utf-8') as file:
            dataset_files = json.loads(file.read())

        # Generate absolute path of files
        self.file_list = []
        dataset_type = DATASET_TYPE_MAPPING[dataset_type]
        samples = dataset_files[dataset_type]

        for s in samples:
            self.file_list.append({
                'sample_name': s,
                'image': cfg.DATASETS.SALICON.IMAGE_PATH % (dataset_type, s),
                'fixation_map': cfg.DATASETS.SALICON.FIXATION_MAP_PATH % (dataset_type, s)
            })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_name = self.file_list[idx]['sample_name']
        image_path = self.file_list[idx]['image']
        fixation_map_path = self.file_list[idx]['fixation_map']

        # load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print('image not found!')
            print(image_path)
            raise
        image = image.astype(np.float32)

        if len(image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (datetime.now(), image_path))
            image = np.stack((image, ) * 3, -1)
        image /= 255.

        # load fixation map image
        fixation_map_image = cv2.imread(fixation_map_path, 0)
        if fixation_map_image is None:
            print('fixation_map_image not found!')
            print(fixation_map_path)
            raise
        fixation_map_image = fixation_map_image.astype(np.float32)
        fixation_map_image /= 255.

        if self.transforms:
            image, fixation_map_image = self.transforms(image, fixation_map_image)

        return sample_name, image, fixation_map_image


class TestDataset(Dataset):

    def __init__(self, cfg, transforms=None):
        # Setup transforms
        self.transforms = transforms
        self.img_height = cfg.CONST.IMG_W
        self.img_width = cfg.CONST.IMG_H

        # Generate absolute path of files
        self.file_list = []
        image_folder = cfg.DATASETS.TEST.INAGE_FOLDER % cfg.DATASET.DATASET_NAME
        images = os.listdir(image_folder)
        for img in images:
            if not img.lower().endswith('.jpg'):
                continue
            sample_name = os.path.splitext(img)[0]
            self.file_list.append({'sample_name': sample_name, 'image': os.path.join(image_folder, img)})

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_name = self.file_list[idx]['sample_name']

        # load image
        image_path = self.file_list[idx]['image']
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if len(image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (datetime.now(), image_path))
            image = np.stack((image, ) * 3, -1)
        image /= 255.
        image = cv2.resize(image, (self.img_height, self.img_width))

        # generate a pseduo fixation_map in order to call transform function
        fixation_map = np.zeros((self.img_height, self.img_width)).astype(np.float32)
        fixation_map = fixation_map[:, :, np.newaxis]

        if self.transforms:
            image, fixation_map = self.transforms(image, fixation_map)

        return sample_name, image


class SaliconDataset_Mode(Dataset):
    """
    salicon dataset, classify processed, diverse and consistent
    diverse: 6167train + 3366val
    consistent: 3833trian + 1634val
    """
    def __init__(self, cfg, params, dataset_type, dataset_mode, transforms=None):

        dataset_type = DATASET_TYPE_MAPPING[dataset_type]     # train or val
        dataset_mode = DATASET_MODE_MAPPING[dataset_mode]     # diverse or consistent
        self.transforms = transforms

        # 整个salicon数据集统计:15000 = train:6167/3833 + val:3366/1634
        with open(params.mode_salicon_dataset_path, 'r') as f:
            dataset = json.load(f)

        self.img_names = dataset[dataset_type][dataset_mode]

        self.file_list = []
        for s in self.img_names:
            self.file_list.append({
                'sample_name': s,
                'image': cfg.DATASETS.SALICON.IMAGE_PATH % (dataset_type, s),
                'fixation_map': cfg.DATASETS.SALICON.FIXATION_MAP_PATH % (dataset_type, s),
            })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_name = self.file_list[idx]['sample_name']
        image_path = self.file_list[idx]['image']
        fixation_map_path = self.file_list[idx]['fixation_map']
        # load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print('image not found!')
            print(image_path)
            raise
        image = image.astype(np.float32)
        if len(image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (datetime.now(), image_path))
            image = np.stack((image, ) * 3, -1)
        image /= 255.

        # load gray fixation map image
        fixation_map_image = cv2.imread(fixation_map_path, 0)
        if fixation_map_image is None:
            print('fixation map_image not found!')
            print(fixation_map_path)
            raise
        fixation_map_image = fixation_map_image.astype(np.float32)
        fixation_map_image /= 255.

        if self.transforms:
            image, fixation_map_image = self.transforms(image, fixation_map_image)

        return sample_name, image, fixation_map_image


class SaliconDS_Topic(Dataset):

    def __init__(self, cfg, params, dataset_type, topic_num, topic_index, transforms=None):

        dataset_type = DATASET_TYPE_MAPPING[dataset_type]     # train or val
        self.transforms = transforms

        # # 整个salicon数据集统计,多个topic,然后用每个topic的数据训练该topic的open salicon模型
        # # topic_num = 5
        # # topic 0: 2748 train + 1462 val, topic 1: 158 train + 57 val, topic 2: 953 train + 446 val,
        # # topic 3: 5894 train + 2947 val, topic 4: 247 train + 88 val
        # dataset_dir = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/work/IJCAI/data/salicon_statistics_5.json'

        # topic_num = 10
        # topic 0: 564 train + 188 val, topic 1: 30 train + 8 val, topic 2: 5818 train + 3044 val,
        # topic 3: 911 train + 468 val, topic 4: 737 train + 441 val, topic 5: 180 train + 72 val,
        # topic 6: 787 train + 347 val, topic 7: 633 train + 287 val, topic 8: 203 train + 81 val,
        # topic 9: 137 train + 64 val

        dataset_path = os.path.join(params.topic_salicon_dataset_dir, 'salicon_statistics_' + str(topic_num) + '.json')
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        self.img_names = dataset[dataset_type][topic_index]

        self.file_list = []
        for s in self.img_names:
            self.file_list.append({
                'sample_name': s,
                'image': cfg.DATASETS.SALICON.IMAGE_PATH % (dataset_type, s),
                'fixation_map': cfg.DATASETS.SALICON.FIXATION_MAP_PATH % (dataset_type, s),
            })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_name = self.file_list[idx]['sample_name']
        image_path = self.file_list[idx]['image']
        fixation_map_path = self.file_list[idx]['fixation_map']
        # load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print('image not found!')
            print(image_path)
            raise
        image = image.astype(np.float32)
        if len(image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (datetime.now(), image_path))
            image = np.stack((image, ) * 3, -1)
        image /= 255.

        # load gray fixation map image
        fixation_map_image = cv2.imread(fixation_map_path, 0)
        if fixation_map_image is None:
            print('fixation map_image not found!')
            print(fixation_map_path)
            raise
        fixation_map_image = fixation_map_image.astype(np.float32)
        fixation_map_image /= 255.

        if self.transforms:
            image, fixation_map_image = self.transforms(image, fixation_map_image)

        return sample_name, image, fixation_map_image


class SaliconDataset_Pseudo(Dataset):
    def __init__(self, cfg, params, dataset_type, transforms=None):
        self.transforms = transforms
        self.dataset_type = DATASET_TYPE_MAPPING[dataset_type]
        self.file_list = []

        if self.dataset_type == 'train':
            """
            # dataset_files = None
            # Load all files of the dataset
            with open(cfg.DATASETS.SALICON.DATASET_FILE_PATH, encoding='utf-8') as file:
                dataset_files = json.loads(file.read())

            train_samples = dataset_files['train']
            train_image = [cfg.DATASETS.SALICON.IMAGE_PATH % ('train', s) for s in train_samples]
            train_fixation_map = [cfg.DATASETS.SALICON.FIXATION_MAP_PATH % ('train', s) for s in train_samples]
            train_map_image = [cfg.DATASETS.SALICON.MAP_IMAGE_PATH % ('train', s) for s in train_samples]

            amount = len(train_samples)
            for i in range(amount):
                self.file_list.append({
                    'sample_name': train_samples[i],
                    'image': train_image[i],
                    'fixation_map': train_fixation_map[i],
                    'map_image': train_map_image[i],
                })

            """
            """
            # pseudo_coco
            with open(params.pseudo_coco_file_path, 'r') as f:
                dataset_files = json.load(f)

            # mattnet
            special_images = params.mattnet_special_images
            # # aws
            # special_images_dir = params.aws_special_images_dir
            # with open(special_images_dir, 'r') as f:
            #     special_images = json.load(f)

            samples = dataset_files['train']

            for s in samples:
                if s in special_images:
                    continue
                self.file_list.append({
                    'sample_name': s,
                    'image': os.path.join('/Shared_Resources/social_media/DataSet/COCO/train2014', s + '.jpg'),
                    'fixation_map': cfg.DATASETS.SALICON.FIXATION_MAP_PATH % (self.dataset_type, s),
                    'map_image': os.path.join(params.pseudo_mattnet_coco_fixation_map_dir, s + '.png'),
                    # 'map_image': os.path.join(params.pseudo_aws_coco_fixation_map_dir, s + '.jpg')
                })
            """

            # combine dataset
            with open(cfg.DATASETS.SALICON.DATASET_FILE_PATH, encoding='utf-8') as file:
                salicon_files = json.loads(file.read())
            salicon_train_samples = salicon_files['train']
            with open(params.pseudo_coco_file_path, 'r') as f:
                pseudo_coco_files = json.load(f)

            pseudo_coco_samples = pseudo_coco_files['train'] + pseudo_coco_files['val']
            # mattnet
            special_images = params.mattnet_special_images
            # # aws
            # special_images_dir = params.aws_special_images_dir
            # with open(special_images_dir, 'r') as f:
            #     special_images = json.load(f)

            for s in pseudo_coco_samples:
                if s in special_images:
                    continue
                sample_name = s
                image = os.path.join(params.coco_image_dir, s + '.jpg')
                fixation_map, map_image = None, None
                if s in salicon_train_samples:
                    fixation_map = cfg.DATASETS.SALICON.FIXATION_MAP_PATH % ('train', s)
                    map_image = cfg.DATASETS.SALICON.MAP_IMAGE_PATH % ('train', s)
                else:
                    fixation_map = cfg.DATASETS.SALICON.FIXATION_MAP_PATH % (self.dataset_type, s)
                    map_image = os.path.join(params.pseudo_mattnet_coco_fixation_map_dir, s + '.png')
                self.file_list.append({
                    'sample_name': sample_name,
                    'image': image,
                    'fixation_map': fixation_map,
                    'map_image': map_image,
                })

        else:
            f = np.load(params.pascal_fixation_map_path)
            pascal_fixation_map = f['fixation_map']
            with open(params.pascal_image_name_path, 'r') as f:
                pascal_image_names = json.load(f)
            amount = len(pascal_image_names)
            for i in range(amount):
                self.file_list.append({
                    'sample_name': pascal_image_names[i],
                    'image': os.path.join(params.pascal_image_dir, pascal_image_names[i] + '.jpg'),
                    'fixation_map': pascal_fixation_map[i],
                    'map_image': pascal_fixation_map[i],
                })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_name = self.file_list[idx]['sample_name']
        image_path = self.file_list[idx]['image']
        fixation_map_path = self.file_list[idx]['fixation_map']

        # load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print('image not found!')
            print(image_path)
            raise
        image = image.astype(np.float32)

        if len(image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (datetime.now(), image_path))
            image = np.stack((image, ) * 3, -1)
        image /= 255.

        # load fixation map image
        fixation_map_image = cv2.imread(fixation_map_path, 0)
        if fixation_map_image is None:
            print('fixation_map_image not found!')
            print(fixation_map_path)
            raise
        fixation_map_image = fixation_map_image.astype(np.float32)
        fixation_map_image /= 255.

        if self.transforms:
            image, fixation_map_image = self.transforms(image, fixation_map_image)

        return sample_name, image, fixation_map_image


def get_base_dataloader():
    print('base salicon data')
    train_data_loader = DataLoader(
        dataset=SaliconDataset(cfg, DatasetType.TRAIN, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True)

    val_data_loader = DataLoader(
        dataset=SaliconDataset(cfg, DatasetType.VAL, val_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=False)

    # test_data_loader = DataLoader(
    #     dataset=TestDataset(cfg),
    #     batch_size=cfg.CONST.BATCH_SIZE,
    #     num_workers=cfg.CONST.NUM_WORKER,
    #     pin_memory=True,
    #     shuffle=False)

    return train_data_loader, val_data_loader


# 加载指定模态的dataloader
def get_mode_dataloader(mode='diverse'):
    print('{} mode salicon data'.format(mode))
    data_mode = None
    if mode == 'diverse':
        data_mode = DatasetMode.Diverse
    elif mode == 'consistent':
        data_mode = DatasetMode.Consistent
    else:
        print('Invalid mode')
        return None, None

    train_data_loader = DataLoader(
        dataset=SaliconDataset_Mode(cfg, params, DatasetType.TRAIN, data_mode, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True)

    val_data_loader = DataLoader(
        dataset=SaliconDataset_Mode(cfg, params, DatasetType.VAL, data_mode, val_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=False)

    return train_data_loader, val_data_loader


# 加载指定topic的dataloader
def get_topic_dataloader(topic_num, topic_index):
    print('topic {}: the {} topic salicon data'.format(topic_num, topic_index))
    train_data_loader = DataLoader(
        dataset=SaliconDS_Topic(cfg, params, DatasetType.TRAIN, topic_num, topic_index, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True)

    val_data_loader = DataLoader(
        dataset=SaliconDS_Topic(cfg, params, DatasetType.VAL, topic_num, topic_index, val_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=False)

    return train_data_loader, val_data_loader


def get_pseudo_coco_dataloader():
    print('pseudo coco fixation map dataset')
    train_data_loader = DataLoader(
        dataset=SaliconDataset_Pseudo(cfg, params, DatasetType.TRAIN, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True)

    val_data_loader = DataLoader(
        dataset=SaliconDataset_Pseudo(cfg, params, DatasetType.VAL, val_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=False)

    return train_data_loader, val_data_loader


def test():
    print('test data loader')
    # # salicon dataset
    # print(10000 + 5000)
    train_data_loader, val_data_loader = get_base_dataloader()

    # # diverse mode + consistent mode
    # print(6167 + 3833)
    # print(3366 + 1634)
    # train_data_loader, val_data_loader = get_mode_dataloader(mode='diverse')

    # # topic num = 5
    # print(2748 + 158 + 953 + 5894 + 247)
    # print(1462 + 57 + 446 + 2947 + 88)

    # # topic num = 10
    # print(564 + 30 + 5818 + 911 + 737 + 180 + 787 + 633 + 203 + 137)
    # print(188 + 8 + 3044 + 468 + 441 + 72 + 347 + 287 + 81 + 64)

    # train_data_loader, val_data_loader = get_topic_dataloader(topic_num=5, topic_index=2)

    print(len(train_data_loader.dataset))
    print(len(val_data_loader.dataset))
    sample_name, sample_img, sample_map_image = next(iter(train_data_loader))
    # sample_name, sample_img, sample_map_image = next(iter(val_data_loader))

    # print(sample_name)
    print(sample_img.size())
    print(sample_map_image.size())


if __name__ == '__main__':
    test()
