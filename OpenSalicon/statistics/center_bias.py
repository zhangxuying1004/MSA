import torch
import numpy as np
import os

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.config import cfg
from OpenSalicon.utils.dataloader import get_pseudo_coco_dataloader


# 计算传入dataloader的center_bias
def calculate_center_bias(data_loader):
    center_bias = torch.zeros(1, cfg.CONST.IMG_H, cfg.CONST.IMG_W)

    for _, _, map_images in data_loader:
        batch_size = map_images.size(0)
        for i in range(batch_size):
            center_bias += map_images[i]

    max_value = torch.max(center_bias)
    center_bias /= max_value

    return center_bias.numpy()


def save_center_bias(center_bias, center_bias_dir, center_bias_name):
    center_bias_path = os.path.join(center_bias_dir, center_bias_name)
    np.savez(center_bias_path, train=center_bias)


def main():
    train_data_loader, _ = get_pseudo_coco_dataloader()
    print(len(train_data_loader.dataset))

    print('start')
    center_bias = calculate_center_bias(train_data_loader)
    center_bias_dir = '/home/zhangxuying/Project/Paper_code/MSA/OpenSalicon/output/data/center_bias'
    # salicon_total
    center_bias_name = 'salicon_total.npz'
    # # mattnet_pseudo_coco_total
    # center_bias_name = 'mattnet_coco_total.npz'
    # # combine salicon_coco
    # center_bias_name = 'mattnet_coco_salicon_combine.npz'
    # # salicon train
    # center_bias_name = 'salicon_train.npz'
    # # mattnet
    # center_bias_name = 'pseudo_coco_train.npz'
    # # aws
    # center_bias_name = 'aws_coco_train.npz'

    save_center_bias(center_bias, center_bias_dir, center_bias_name)
    print('finished!')


def test():
    train_dataloader, _ = get_pseudo_coco_dataloader()
    print(len(train_dataloader.dataset))
    _, _, sample_mpas = next(iter(train_dataloader))
    print(sample_mpas.size())


if __name__ == "__main__":
    main()
    # test()
# json 文件需要是序列化的数据
