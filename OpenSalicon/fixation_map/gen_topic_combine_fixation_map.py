import torch
from torch.nn import functional as F
import torchvision.transforms as T
from pathlib import Path
import os
import numpy as np
import cv2

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.salicon import Salicon
from OpenSalicon.utils.config import cfg
from topic_classifier.classifier import Classifier


def load_classifier_model(topic_classifier_output_dir, topic_num):
    topic_classifier_log_dir = os.path.join(topic_classifier_output_dir, 'model_logs', str(topic_num))
    topic_classifier_model_dir = os.path.join(topic_classifier_output_dir, 'model', str(topic_num))

    model_log_path = os.path.join(topic_classifier_log_dir, 'cider_log_' + str(topic_num) + '.txt')
    model_log = np.loadtxt(model_log_path)
    index = np.argmax(model_log[:, 1].tolist())

    checkpoint = os.path.join(topic_classifier_model_dir, str(index) + '.pkl')

    classifier_model = Classifier()
    classifier_model.load_state_dict(torch.load(checkpoint))
    classifier_model = classifier_model.cuda()
    classifier_model.eval()
    return classifier_model


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


def load_topic_center_bias(center_bias_dir, topic_num):
    topic_center_bias = []
    # eps = 1e-6
    for topic_index in range(topic_num):
        topic_index_center_bias_path = os.path.join(center_bias_dir + str(topic_num), 'topic_' + str(topic_index) + '.npz')
        topic_index_center_bias = np.load(topic_index_center_bias_path)['train'] * 255.

        # topic_index_center_bias = np.log(topic_index_center_bias + eps)
        # topic_index_center_bias -= np.min(topic_index_center_bias)

        topic_index_center_bias = normalize_numpy_data(np.squeeze(topic_index_center_bias))
        topic_center_bias.append(topic_index_center_bias)

    return topic_center_bias


def load_model(topic_model_path, topic_num):
    assert len(topic_model_path) == topic_num
    topic_model = []
    for topic_index in range(topic_num):
        # topic_index
        topic_index_model = Salicon().cuda()
        if torch.cuda.is_available():
            topic_index_model = torch.nn.DataParallel(topic_index_model).cuda()
        checkpoint_index = torch.load(topic_model_path[topic_index])
        topic_index_model.load_state_dict(checkpoint_index['model_state_dict'])
        topic_index_model.eval()
        topic_model.append(topic_index_model)
    return topic_model


def main():
    datasets = ['bruce', 'judd', 'pascal']
    for dataset in datasets:
        img_path = '/home/data/XSSUN/datasets/imgs/{}'.format(dataset)  # bruce, judd, pascal for different dataset

        output_path = '/home/data/XSSUN/algmaps/{}/Topic_5_Combine_blur/'.format(dataset)
        # output_path = '/home/data/XSSUN/algmaps/{}/Topic_10_Combine_blur/'.format(dataset)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # load model
        topic_num = 5
        topic_model_path = [
            '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T21-06-07.324318/best-ckpt.pth.tar',
            '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T21-22-23.131735/best-ckpt.pth.tar',
            '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T21-24-32.742317/best-ckpt.pth.tar',
            '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T22-36-42.945320/best-ckpt.pth.tar',
            '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T23-08-19.583680/best-ckpt.pth.tar',
        ]

        # topic_num = 10
        # topic_model_path = [
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T23-13-51.990732/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T23-21-36.456375/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-01T23-31-30.787974/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-02T09-27-17.084694/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-02T09-34-37.268985/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-02T09-42-06.644884/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-02T09-45-27.951790/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-02T09-52-23.528646/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-02T09-58-03.956828/best-ckpt.pth.tar',
        #     '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/topic_output/checkpoints/2020-03-02T10-00-25.400901/best-ckpt.pth.tar',
        # ]

        # 加载训练好的opensalicon模型
        topic_model = load_model(topic_model_path, topic_num)

        # 加载分类器模型
        topic_classifier_output_dir = '/home/zhangxuying/Project/Paper_code/MSA/topic_classifier/output/'
        classifier_model = load_classifier_model(topic_classifier_output_dir, topic_num)
        classifier_model.eval()

        # # center bias，log
        center_bias_dir = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/work/IJCAI/data/center_bias/topics/'
        topic_center_bias = load_topic_center_bias(center_bias_dir, topic_num)

        # img preprocess
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        ])

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

                topic_saliency_map = []
                for topic_index in range(topic_num):
                    # topic_index
                    topic_index_saliency_map = gen_fixation_map(img, topic_model[topic_index])
                    topic_index_saliency_map = normalize_numpy_data(topic_index_saliency_map)

                    topic_index_saliency_map = topic_index_saliency_map * topic_center_bias[topic_index]

                    topic_saliency_map.append(topic_index_saliency_map)

                # 混合参数
                # logits = classifier_model(img).cpu().numpy()
                # alpha = F.softmax(logits)
                alpha = F.softmax(classifier_model(img), dim=1).squeeze().cpu().numpy()
                # print('predict:', alpha)

                # 混合
                # saliency_map = alpha[0] * topic_0_saliency_map + alpha[1] * topic_1_saliency_map + alpha[2] * topic_2_saliency_map + alpha[3] * topic_3_saliency_map + alpha[4] * topic_4_saliency_map
                saliency_map = np.zeros_like(topic_saliency_map[0])
                for topic_index in range(topic_num):
                    saliency_map = saliency_map + alpha[topic_index] * topic_saliency_map[topic_index]

                # 0~1化处理，除以最大值
                saliency_map /= np.max(saliency_map)
                saliency_map *= 255.

                # 高斯模糊
                kernel_size = (31, 31)
                saliency_map = cv2.GaussianBlur(saliency_map, kernel_size, 0)

                # 保存
                cv2.imwrite(os.path.join(output_path, filename.name.split('.')[0] + '.png'), saliency_map)
                print('{}/'.format(i))

        print('Done')


if __name__ == '__main__':
    main()
    # test()
