import torch
from torch import nn
from torchvision.models import vgg16
import sys
sys.path.append('/home/zxy/Project/IJCAI/')
from binary_classification.config import Flatten


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        base_model = vgg16(pretrained=True)

        self.net = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),

            nn.Dropout(0.5),
            nn.Linear(512, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.net(x)
        x = x.squeeze(dim=-1)

        return x


def test():

    model = Classifier()
    x = torch.randn(64, 3, 224, 224)

    out = model(x)
    print(x.size(), out.size())
    print(torch.min(out).item(), torch.max(out).item())


if __name__ == '__main__':
    test()
