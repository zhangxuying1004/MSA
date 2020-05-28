import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from time import time
# import json

from config import Parameters
from dataset import Sub_COCO
from classifier import Classifier


def load_model(params):
    print('reading logs')
    log_path = os.path.join(params.model_logs_dir, 'cider_log_' + str(params.topic_num) + '.txt')
    assert os.path.exists(log_path)
    log = np.loadtxt(log_path)
    index = np.argmax(log[:, 1].tolist())

    print('loading model')
    checkpoint = os.path.join(params.model_dir, str(index) + '.pkl')
    assert os.path.exists(checkpoint)

    model = Classifier()
    model.load_state_dict(torch.load(checkpoint))
    return model


def main():

    t1 = time()
    # load hyperparams
    params = Parameters()

    # load dataset
    print('loading dataset')
    dataset = Sub_COCO(params)
    print('data num:{}'.format(len(dataset)))
    # x:[b, 3, 224, 224]
    # y:[b,]
    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False
    )

    # load model
    model = load_model(params).cuda().eval()

    # run model
    correct_num = 0.0
    total_num = 0.0

    print('starting')
    for (x, y) in dataloader:
        x, y = x.cuda(), y.cuda()

        logits = model(x)
        correct_num += y.int().eq(torch.argmax(logits, dim=1)).sum().float()
        total_num += y.size(0)
    accuracy = correct_num / total_num
    print(accuracy)
    t2 = time()
    print('it costs {} s'.format(t2 - t1))


def test():
    params = Parameters()
    print('reading logs')
    log_path = os.path.join(params.model_logs_dir, 'cider_log_' + str(params.topic_num) + '.txt')
    print(os.path.exists(log_path))
    log = np.loadtxt(log_path)
    index = np.argmax(log[:, 1].tolist())

    print('loading model')
    checkpoint = os.path.join(params.model_dir, str(index) + '.pkl')
    print(os.path.exists(checkpoint))


if __name__ == "__main__":
    # test()
    main()

# topic_5, test accuracy = 0.8789
# topic_10, test accuracy = 0.8081
