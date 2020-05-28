from torch import nn
import os


class Parameters:
    def __init__(self):
        # dataset
        self.coco_image_dir = '/home/zhangxuying/DataSet/COCO/train2014/'
        self.dataset_dir = '/home/zhangxuying/Project/Paper_code/MSA/dataset_constructor/output/'
        self.dataset_mode = 'train'

        # model
        self.model_logs_dir = '/home/zhangxuying/Project/Paper_code/MSA/mode_classifier/output/model_logs/'
        self.model_dir = '/home/zhangxuying/Project/Paper_code/MSA/mode_classifier/output/cider_saved_models/'

        # hyperparameters
        self.epochs = 25
        self.batch_size = 64
        self.learning_rate = 3e-4


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def test():
    config = Parameters()

    print(os.path.exists(config.coco_info_dir))
    print(os.path.exists(config.dataset_dir))

    # print(config.model_logs)
    # print(os.path.exists(config.model_logs))
    #
    # print(config.model_path)
    # print(os.path.exists(config.model_path))
    #
    # flatten = Flatten()
    # x = torch.randn(6, 512, 7, 7)
    # print(x.size())
    # x = flatten(x)
    # print(x.size())


if __name__ == '__main__':
    test()
