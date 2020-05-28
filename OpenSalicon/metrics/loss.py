import torch
from torch import nn
import numpy as np
import sys
sys.path.append('/Shared_Resources/social_media/project/zxy/OpenSalicon/')

from utils.config import cfg
from metrics.kl_divergence import KLDivergence
from time import time


class MyBCELOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyBCELOSS, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):

        input = input.view(-1)
        target = target.view(-1)

        loss = -(target * torch.log(input) + (1 - target) * torch.log(1 - input))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError

        return loss


class MyMSELOSS(nn.Module):
    def __init__(self):
        super(MyMSELOSS, self).__init__()

    def forward(self, input, target):
        input = input.view(1, -1)
        target = target.view(1, -1)

        num = target.size(1)
        item = target - input

        loss = item.mm(item.t()) / num
        loss = loss.squeeze()
        # loss.requires_grad = True
        return loss


class MyKLDLOSS(nn.Module):
    def __init__(self):
        super(MyKLDLOSS, self).__init__()
        self.kl_divergence = KLDivergence(cfg.NETWORK.EPS)

    def forward(self, input, target):
        return self.kl_divergence(input, target)


def test():
    y_true = [0, 0, 1, 1]
    y_pred = [.4, .6, .3, .7]
    # y_true = torch.from_numpy(np.array(y_true))
    y_true = torch.tensor(y_true)
    y_true = y_true.view(2, 2)
    y_pred = torch.tensor(y_pred)
    y_pred = y_pred.view(2, 2)
    print(y_pred)
    print(y_true)
    criterion = MyBCELOSS()
    loss = criterion(y_pred, y_true)
    print(loss)


if __name__ == '__main__':
    np.random.seed(123)
    # test()
    # a = np.log(0.6) + np.log(0.4) + np.log(0.3) + np.log(0.7)
    # print(-a/4)

    y_true = torch.rand(1, 1, 224, 224)
    y_pred = torch.rand(1, 1, 224, 224)

    t1 = time()
    criterion = MyBCELOSS()
    loss = criterion(y_pred, y_true)
    print(loss)
    t2 = time()
    print('bce loss costs {} s'.format(t2 - t1))

    t3 = time()
    criterion = MyMSELOSS()
    loss = criterion(y_pred, y_true)
    print(loss)
    t4 = time()
    print('mse loss costs {} s'.format(t4 - t3))

    t5 = time()
    criterion = MyKLDLOSS()
    loss = criterion(y_pred, y_true)
    print(loss)
    t6 = time()
    print('kld loss costs {} s'.format(t6 - t5))
