import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import json

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from topic_classifier.utils.config import Parameters


class Sub_COCO(Dataset):
    def __init__(self, params):
        super(Sub_COCO, self).__init__()
        self.coco_image_dir = params.coco_image_dir
        self.topic_num = params.topic_num
        self.dataset_path = os.path.join(params.dataset_dir, 'dataset_' + str(self.topic_num) + '.json')

        self.mode = params.dataset_mode
        self.load()

    def load(self):

        with open(self.dataset_path, 'r') as f:
            dataset_file = json.load(f)

        dataset = dataset_file[self.mode]
        self.image_names = dataset['image_name']
        self.labels = dataset['label']

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]

        img_tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((224, 224)),
            # 数据值从[0,255]范围转为[0,1]，相当于除以255.操作
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = img_tf(os.path.join(self.coco_image_dir, image_name))
        # print(label)
        # label = one_hot(torch.tensor(label).view(-1), depth=self.topic_num).squeeze()
        label = torch.tensor(label)

        return img, label


def test():
    print('sub coco data')
    params = Parameters()
    dataset = Sub_COCO(params)

    print('{} data_num：{}'.format(params.dataset_mode, len(dataset)))

    data_loader = DataLoader(
        dataset,
        batch_size=3
    )

    sample_x, sample_y = next(iter(data_loader))
    print(sample_x.shape, sample_y.shape)
    print(sample_y)


if __name__ == '__main__':
    test()
