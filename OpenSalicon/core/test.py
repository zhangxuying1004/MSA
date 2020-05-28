import torch
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import cv2
import os

import sys
sys.path.append('/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/')

# from utils.config import cfg
from utils import data_transforms, network_utils
from utils.dataloader import TestDataset
from salicon import Salicon


def test_net(cfg):
    # load dataset
    test_transforms = data_transforms.Compose([
        data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        data_transforms.ToTensor(),
    ])
    test_data_loader = DataLoader(
        dataset=TestDataset(cfg, test_transforms),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False
    )

    # load model
    model = Salicon()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    print('[INFO] %s Loading weights from %s ...' % (datetime.now(), cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create folder to save predicted results
    output_dir = os.path.join(cfg.DIR.OUT_PATH, 'images', cfg.DATASET.DATASET_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    n_samples = len(test_data_loader)
    for batch_idx, (sample_name, image) in enumerate(test_data_loader):
        sample_name = sample_name[0] if isinstance(sample_name[0], str) else sample_name[0].item()

        with torch.no_grad():
            image = network_utils.var_or_cuda(image)
            saliency_map = model(image)
            saliency_map = saliency_map.squeeze().cpu().numpy()

            cv2.imwrite(os.path.join(output_dir, '%s.png' % sample_name), saliency_map / np.max(saliency_map) * 255)
            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Predicting Sample = %s' % (datetime.now(), batch_idx + 1, n_samples, sample_name))


if __name__ == '__main__':
    test_net()
