import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from time import time

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from mode_classifier.utils.config import Parameters
from mode_classifier.utils.dataset import Sub_COCO
from mode_classifier.classifier import Classifier


def load_model(params):
    # load model，在val数据集上performance最好的模型
    log_path = params.model_logs_dir + 'cider_log.txt'
    assert os.path.exists(log_path)

    log_ = np.loadtxt(log_path)
    index = np.argmax(log_[:, 4].tolist())
    epoch, batch_idx = int(log_[index][0]), int(log_[index][1])
    print(epoch, batch_idx)

    checkpoint_path = params.model_dir + str(epoch) + '_' + str(batch_idx) + '.pkl'
    assert os.path.exists(checkpoint_path)

    model = Classifier()
    model.load_state_dict(torch.load(checkpoint_path))

    return model


def main():
    t1 = time()
    params = Parameters()

    # load dataset
    dataset = Sub_COCO(params, mode='test')
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
    for (x, y) in dataloader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        preds = (logits + 0.5).int()
        correct_num += preds.eq(y.int()).sum().float()
        total_num += y.size(0)
    accuracy = correct_num / total_num
    print(accuracy)

    t2 = time()
    print('it costs {} s'.format(t2 - t1))


if __name__ == "__main__":
    main()
