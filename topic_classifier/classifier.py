import torch
from torch import nn
from torchvision.models import vgg16

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from topic_classifier.utils.config import Flatten, Parameters
params = Parameters()


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        base_model = vgg16(pretrained=True)
        # print(base_model)

        self.net = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),

            nn.Dropout(0.5),
            nn.Linear(512, params.topic_num, bias=True),

        )

    def forward(self, x):

        logits = self.net(x)
        # x = x.squeeze(dim=-1)

        return logits


def test():

    model = Classifier()
    x = torch.randn(3, 3, 224, 224)

    out = model(x)
    print(x.size(), out.size())
    # print(torch.min(out).item(), torch.max(out).item())


if __name__ == '__main__':
    test()
