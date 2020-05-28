from torch import nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        pos_mask = target.gt(0).float()
        neg_mask = target.eq(0).float()

        pos = input * pos_mask
        neg = input * neg_mask

        pos_loss = F.binary_cross_entropy(pos, target)
        neg_loss = F.binary_cross_entropy(neg, target)

        return pos_loss * self.pos_weight + neg_loss
