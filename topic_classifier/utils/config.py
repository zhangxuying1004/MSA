import torch
from torch import nn
import os


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth).long()
    idx = torch.unsqueeze(label, dim=1)
    out.scatter_(dim=1, index=idx, value=1)
    return out.float()


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Parameters:
    def __init__(self):
        self.coco_image_dir = '/home/zhangxuying/DataSet/COCO/train2014/'

        self.topic_num = 10
        # dataset
        self.dataset_dir = '/home/zhangxuying/Project/Paper_code/MSA/dataset_constructor/output/'
        self.dataset_mode = 'train'

        # model
        self.model_dir = '/home/zhangxuying/Project/Paper_code/MSA/topic_classifier/output/saved_models/' + str(self.topic_num) + '/'
        self.model_logs_dir = '/home/zhangxuying/Project/Paper_code/MSA/topic_classifier/output/model_logs/' + str(self.topic_num) + '/'

        # hyperparams
        self.epochs = 20
        self.batch_size = 64
        self.learning_rate = 1e-4


def test():
    params = Parameters()

    print(os.path.exists(params.coco_image_dir))

    print(os.path.exists(params.dataset_path))

    label = torch.tensor([3, 2, 4, 0, 1, 3])
    label = one_hot(label, depth=5)
    print(label)
    print(label.shape)
    print(params.dataset_mode)


if __name__ == '__main__':
    test()
