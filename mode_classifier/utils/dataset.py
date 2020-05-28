import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import json

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA')
from mode_classifier.utils.config import Parameters


class Sub_COCO(Dataset):
    def __init__(self, params, mode='train'):
        super(Sub_COCO, self).__init__()
        assert os.path.exists(params.dataset_dir)

        self.dataset_path = os.path.join(params.dataset_dir, 'dataset_mode_cider.json')
        self.mode = mode
        self.load(params)

    def load(self, params):
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        dataset = dataset[self.mode]
        self.imgs_path = [os.path.join(params.coco_image_dir, img_name) for img_name in dataset['image_name']]
        self.labels = dataset['label']

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        label = self.labels[idx]
        # 将图片路径映射成图片变量
        img_tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((224, 224)),
            # 数据值从[0,255]范围转为[0,1]，相当于除以255.操作
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = img_tf(img_path)
        label = torch.tensor(label)

        return img, label


def test():
    print('sub coco data')
    params = Parameters()

    dataset = Sub_COCO(params, mode='test')
    print('data_num：', len(dataset))

    data_loader = DataLoader(
        dataset,
        batch_size=3
    )

    sample_x, sample_y = next(iter(data_loader))
    print(sample_x.shape, sample_y.shape)


if __name__ == '__main__':
    test()
