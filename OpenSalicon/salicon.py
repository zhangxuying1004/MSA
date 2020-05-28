import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.config import cfg
from OpenSalicon.utils.network_utils import normalize_map


class Salicon(nn.Module):
    def __init__(self):
        super(Salicon, self).__init__()

        self.fine_sampling = nn.Upsample(size=(2 * cfg.CONST.IMG_H, 2 * cfg.CONST.IMG_W))
        # self.coarse_sampling = nn.Upsample(size=(600, 800))

        self.dnn_seq = vgg16(pretrained=True).features
        # print(self.dnn_seq)

        # self.interpolation = nn.Upsample(size=(28, 28))
        self.interpolation = nn.Upsample(size=(int(cfg.CONST.IMG_H / 16), int(cfg.CONST.IMG_W / 16)))

        self.integration = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: (b, 3, 224, 224)
        :return: saliency map:(b, 1, 224, 224)
        """

        coarse_img = x
        fine_img = self.fine_sampling(x)

        # (b, 512, 7, 7) => (b, 512, 14, 14)
        coarse_feats = self.dnn_seq(coarse_img)

        coarse_feats = self.interpolation(coarse_feats)
        # (b, 512, 14, 14)
        fine_feats = self.dnn_seq(fine_img)

        feats = torch.cat((coarse_feats, fine_feats), dim=1)

        saliency_map = self.integration(feats)
        saliency_map = normalize_map(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(cfg.CONST.IMG_H, cfg.CONST.IMG_W))
        return saliency_map


def test():
    x = torch.randn(64, 3, 224, 224)
    model = Salicon()
    out = model(x)

    print(out.size())


if __name__ == '__main__':
    test()

    # x = torch.randn(1, 3, 448, 448)
    # y = torch.randn(1, 3, 2 * 448, 2 * 448)
    # model = vgg16(pretrained=False).features
    #
    # out1 = model(x)
    # out2 = model(y)
    # print(out1.size())
    # print(out2.size())
