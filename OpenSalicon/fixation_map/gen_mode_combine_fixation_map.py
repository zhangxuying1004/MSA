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
from mode_classifier.classifier import Classifier


# img preprocess
transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
])


# 加载分类器模型
def load_classifier_model():
    mode_classifier_log_dir = '/home/zhangxuying/Project/Paper_code/MSA/mode_classifier/output/model_logs'
    model_log_path = os.path.join(mode_classifier_log_dir, 'cider_log.txt')

    model_logs = np.loadtxt(model_log_path)
    index = np.argmax(model_logs[:, 4].tolist())
    epoch, batch_idx = int(model_logs[index][0]), int(model_logs[index][1])

    mode_classifier_model_dir = '/home/zhangxuying/Project/Paper_code/MSA/mode_classifier/output/cider_saved_models'
    checkpoint = os.path.join(mode_classifier_model_dir, str(epoch) + '_' + str(batch_idx) + '.pkl')

    classifier_model = Classifier()
    classifier_model.load_state_dict(torch.load(checkpoint))
    classifier_model = classifier_model.cuda()
    classifier_model.eval()
    return classifier_model


# load open salicon model in mode
def load_salicon_model(model_path):
    model = Salicon().cuda()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


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


def main():
    datasets = ['bruce', 'judd', 'pascal']
    for dataset in datasets:
        img_path = '/home/data/XSSUN/datasets/imgs/{}'.format(dataset)  # bruce, judd, pascal for different dataset
        output_path = '/home/data/XSSUN/algmaps/{}/IJCAI_Combine_Norm_log_center_bias_blur31/'.format(dataset)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # 加载训练好的diverse和consistent(non-diverse) open salicon model
        diverse_model_path = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/ijcai_output/checkpoints/2020-01-10T22-17-15.411937/best-ckpt.pth.tar'
        consistent_model_path = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/ijcai_output/checkpoints/2020-01-10T22-27-23.496069/best-ckpt.pth.tar'
        # diverse
        diverse_model = load_salicon_model(diverse_model_path)
        # consistent
        consistent_model = load_salicon_model(consistent_model_path)

        # 加载分类器模型
        classifier_model = load_classifier_model()

        # 加载center bias，
        center_bias_dir = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/work/IJCAI/data/center_bias/files'
        diverse_center_bias_path = os.path.join(center_bias_dir, 'diverse_center_bias.npz')
        consistent_center_bias_path = os.path.join(center_bias_dir, 'consistent_center_bias')

        diverse_center_bias = load_center_bias(diverse_center_bias_path)
        consistent_center_bias = load_center_bias(consistent_center_bias_path)

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

                # 生成diverse fixation map
                diverse_saliency_map = gen_fixation_map(img, diverse_model)
                # diverse fixation map加归一化
                diverse_saliency_map = normalize_numpy_data(diverse_saliency_map)
                # 加 center bias
                diverse_saliency_map = diverse_saliency_map * diverse_center_bias

                # 生成consistent fixation map
                consistent_saliency_map = gen_fixation_map(img, consistent_model)
                # consistent fixation map加归一化
                consistent_saliency_map = normalize_numpy_data(consistent_saliency_map)
                # 加 center bias
                consistent_saliency_map = consistent_saliency_map * consistent_center_bias

                # 模态混合
                alpha = classifier_model(img).cpu().numpy()[0]
                # print('predict:', alpha)
                saliency_map = cv2.addWeighted(consistent_saliency_map, 1 - alpha, diverse_saliency_map, alpha, 0)

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
