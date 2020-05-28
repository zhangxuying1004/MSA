import torch
import os
import numpy as np
import json
from time import time

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from mode_classifier.classifier import Classifier
from OpenSalicon.utils.dataloader import get_base_dataloader
from OpenSalicon.utils.config import cfg


def load_model():
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


def statistical(data_loader, class_model):
    diverse = []
    consistent = []
    for (sample_names, images, fixation_maps) in data_loader:
            images = images.cuda()
            score = class_model(images)
            predict = (score + 0.5).int()
            predict = predict.cpu().numpy().tolist()

            for i in range(cfg.CONST.BATCH_SIZE):
                if predict[i] == 0:
                    consistent.append(sample_names[i])
                elif predict[i] == 1:
                    diverse.append(sample_names[i])

    return diverse, consistent


def main():
    start_time = time()
    # 加载数据集
    train_data_loader, val_data_loader = get_base_dataloader()

    # 加载分类器模型
    classifier_model = load_model()

    train_diverse = []
    train_consistent = []

    val_diverse = []
    val_consistent = []

    print('开始统计')
    with torch.no_grad():
        t1 = time()
        # 训练集
        train_diverse, train_consistent = statistical(train_data_loader, classifier_model)
        print('训练集统计完成！')
        print('train_1_num:', len(train_diverse), 'train_0_num:', len(train_consistent))
        t2 = time()
        print('train dataset cost {} s'.format(t2 - t1))

        # 验证集
        t3 = time()
        val_diverse, val_consistent = statistical(val_data_loader, classifier_model)
        print('验证集统计完成！')
        print('val_1_num:', len(val_diverse), 'val_0_num:', len(val_consistent))
        t4 = time()
        print('val dataset cost {} s'.format(t4 - t3))
        train_statistics = {
            'diverse': train_diverse,
            'consistent': train_consistent
        }
        val_statistics = {
            'diverse': val_diverse,
            'consistent': val_consistent
        }

        salicon_statistics = {
            'train': train_statistics,
            'val': val_statistics
        }
        dataset_file_dir = '/home/zhangxuying/Project/Paper_code/MSA/OpenSalicon/output/data'
        dataset_file_path = os.path.join(dataset_file_dir, 'salicon_statistics.json')

        with open(dataset_file_path, 'w') as f:
            json.dump(salicon_statistics, f)

        print('统计内容已保存！')
        end_time = time()
        print('it takes {} s'.format(end_time - start_time))


def test():
    dataset_file_dir = '/home/zhangxuying/Project/Paper_code/MSA/OpenSalicon/output/data'
    dataset_file_path = os.path.join(dataset_file_dir, 'salicon_statistics.json')
    with open(dataset_file_path, 'r') as f:
        salicon_statistics = json.load(f)
    train_statistics = salicon_statistics['train']
    val_statistics = salicon_statistics['val']

    train_diverse = train_statistics['diverse']
    train_consistent = train_statistics['consistent']

    val_diverse = val_statistics['diverse']
    val_consistent = val_statistics['consistent']

    print(len(train_diverse))
    print(len(train_consistent))
    print(len(val_diverse))
    print(len(val_consistent))

    # 6167
    # 3833
    # 3366
    # 1634


if __name__ == '__main__':
    # main()
    test()
