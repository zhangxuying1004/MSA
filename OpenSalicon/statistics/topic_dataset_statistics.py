import torch
import json
import os
import numpy as np

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.dataloader import get_base_dataloader
from topic_classifier.classifier import Classifier


"""
统计多个topic的salicon数据集
"""


def load_classifier_model(topic_num):
    topic_classifier_log_dir = '/home/zhangxuying/Project/Paper_code/MSA/topic_classifier/output/model_logs/' + str(topic_num)
    model_log_path = os.path.join(topic_classifier_log_dir, 'cider_log_' + str(topic_num) + '.txt')
    model_log = np.loadtxt(model_log_path)
    index = np.argmax(model_log[:, 1].tolist())

    topic_classifier_model_dir = '/home/zhangxuying/Project/Paper_code/MSA/topic_classifier/output/model/' + str(topic_num)
    checkpoint = os.path.join(topic_classifier_model_dir, str(index) + '.pkl')

    classifier_model = Classifier()
    classifier_model.load_state_dict(torch.load(checkpoint))
    classifier_model = classifier_model.cuda()
    classifier_model.eval()
    return classifier_model


def statistical(data_loader, classifier_model, topic_num):

    sample_name_list = []
    for i in range(topic_num):
        sample_name_list.append([])
    # print('sample_name list:', sample_name_list)

    for (sample_names, images, fixation_maps) in data_loader:
        images = images.cuda()
        logits = classifier_model(images)
        predict = torch.argmax(logits, dim=1)
        predict = predict.cpu().numpy().tolist()

        for i in range(len(predict)):
            sample_name_list[predict[i]].append(sample_names[i])

    return sample_name_list


def main():
    print('加载数据')
    train_data_loader, val_data_loader = get_base_dataloader()
    print(len(train_data_loader.dataset))
    print(len(val_data_loader.dataset))

    topic_num = 10      # 设置topic的数目,此处应与classification中的topic_num保持一致
    print('加载模型')
    classifier_model = load_classifier_model(topic_num)

    print('统计数据')
    train_sample_name_list = statistical(train_data_loader, classifier_model, topic_num)
    val_sample_name_list = statistical(val_data_loader, classifier_model, topic_num)

    print('保存数据')
    salicon_statistics = {
        'train': train_sample_name_list,
        'val': val_sample_name_list
    }
    dataset_file_dir = '/home/zhangxuying/Project/Paper_code/MSA/OpenSalicon/output/data'
    dataset_file_path = os.path.join(dataset_file_dir, 'salicon_statistics_' + str(topic_num) + '.json')
    with open(dataset_file_path, 'w') as f:
        json.dump(salicon_statistics, f)
    print('finished!')


def test():
    dataset_file_dir = '/home/zhangxuying/Project/Paper_code/MSA/OpenSalicon/output/data'
    with open(os.path.join(dataset_file_dir, 'salicon_statistics_10.json'), 'r') as f:
        salicon_statistics = json.load(f)
    print(salicon_statistics.keys())
    train_sample_name_list, val_sample_name_list = salicon_statistics['train'], salicon_statistics['val']
    print(type(train_sample_name_list))
    print(type(val_sample_name_list))
    print(len(train_sample_name_list))
    print(len(val_sample_name_list))
    print(train_sample_name_list[0])


if __name__ == "__main__":
    # main()
    test()
    # load_model(topic_num=10)
