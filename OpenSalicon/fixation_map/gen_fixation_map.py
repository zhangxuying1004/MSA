import torch
import os
import numpy as np
import cv2
import torchvision.transforms as T
from pathlib import Path

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.salicon import Salicon
from OpenSalicon.utils.config import cfg


# img preprocess
transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
])


def gen_fixation_map(img, open_salicon_model):
    saliency_map = open_salicon_model(img)
    saliency_map = saliency_map.squeeze()
    # normalize
    saliency_map -= torch.min(saliency_map)
    saliency_map /= torch.max(saliency_map)
    saliency_map = saliency_map.cpu().numpy()
    assert np.max(saliency_map) == 1
    assert np.min(saliency_map) == 0
    saliency_map = (saliency_map * 255).astype('uint8')
    return saliency_map


def normalize_numpy_data(data):
    data = np.array(data, dtype=np.float32)
    sum_value = np.sum(data)
    data /= sum_value
    return data


def load_center_bias(center_bias_path):
    # center bias，只使用训练集的center bias，将像素值设置成0~255
    center_bias = np.load(center_bias_path)['train'] * 255.

    # log 处理，并减去最小值
    center_bias = np.log(center_bias)
    center_bias -= np.min(center_bias)
    # 归一化处理，概率形式
    center_bias = normalize_numpy_data(np.squeeze(center_bias))

    return center_bias


def load_model(model_path):
    model = Salicon().cuda()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():
    datasets = ['bruce', 'judd', 'pascal']
    for dataset in datasets:
        img_path = '/home/data/XSSUN/datasets/imgs/{}'.format(dataset)  # bruce, judd, pascal for different dataset
        output_path = '/home/data/XSSUN/algmaps/{}/Combine_mattnet_salicon_from_scratch_blur31'.format(dataset)   # 10 epoch

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # model
        model_path = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/pseudo_coco_output/checkpoints/2020-05-21T00-10-44.459574/best-ckpt.pth.tar'
        model = load_model(model_path)
        # center_bias
        center_bias_path = os.path.join('/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/work/IJCAI/data/center_bias/combine_image', 'combine_mattnet_coco.npz')
        center_bias = load_center_bias(center_bias_path)

        with torch.no_grad():
            for i, filename in enumerate(Path(img_path).iterdir()):
                img = cv2.imread(str(filename))
                if img is None:
                    print('failed to read {}'.format(str(filename)))
                    if filename.name.split('.')[-1] != '.jpg':
                        continue
                    else:
                        break
                img = transforms(img).cuda().unsqueeze(0)

                saliency_map = gen_fixation_map(img, model)
                saliency_map = normalize_numpy_data(saliency_map)
                saliency_map = saliency_map * center_bias

                # 0~1化处理，除以最大值
                saliency_map /= np.max(saliency_map)
                saliency_map *= 255.

                # 高斯模糊
                kernel_size = (31, 31)
                saliency_map = cv2.GaussianBlur(saliency_map, kernel_size, 0)

                cv2.imwrite(os.path.join(output_path, filename.name.split('.')[0] + '.png'), saliency_map)
                print('{}/'.format(i))

        print('Done')


if __name__ == '__main__':
    main()
