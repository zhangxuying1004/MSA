import json
import numpy as np
import os
import random

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from caption_metrics.pycocoevalcap.cider.cider import Cider
from caption_metrics.pycocoevalcap.rouge.rouge import Rouge
# from caption_metrics.pycocoevalcap.bleu.bleu import Bleu

from utils import Parameters


# 剔除句子末尾的句号
def preprocess(string):
    word_list = word_tokenize(string)
    if word_list[-1] == '.':
        word_list = word_list[:-1]
    return word_list


# 将单词列表转化为字符串
def reprocess(word_list):
    string = ''
    for w_item in word_list:
        string += (w_item + ' ')
    string = string[:-1]
    return string


# 计算BLEU得分
def bleu_score(candidate, reference):
    candidate = preprocess(candidate)
    reference = [preprocess(reference)]
    score = sentence_bleu(reference, candidate)

    # score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    # score = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    # score = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    # score = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
    return score


# def bleu_score(candidate, reference):
#     id = random.randint(0, 10)
#     candidate = {str(id): [reprocess(preprocess(candidate))]}
#     reference = {str(id): [reprocess(preprocess(reference))]}
#
#     scorer = Bleu()
#     (score, scores) = scorer.compute_score(reference, candifwdate)
#     return score[0]

def cider_score(candidate, reference):
    id = random.randint(0, 10)
    candidate = {str(id): [reprocess(preprocess(candidate))]}
    reference = {str(id): [reprocess(preprocess(reference))]}

    scorer = Cider()
    (score, scores) = scorer.compute_score(reference, candidate)
    return score


def rouge_score(candidate, reference):
    id = random.randint(0, 10)
    candidate = {str(id): [reprocess(preprocess(candidate))]}
    reference = {str(id): [reprocess(preprocess(reference))]}

    scorer = Rouge()
    (score, scores) = scorer.compute_score(reference, candidate)
    return score


def meteor_score(candidate, reference):
    pass


def spice_score(candidate, reference):
    pass


def calculate_similarity(captions, num=5, mode='cider'):
    # captions = captions[:num]
    similarity_matrix = np.ones([num, num])

    for i in range(num):
        for j in range(num):
            score = 0.
            candidate = captions[i]
            reference = captions[j]
            if mode == 'bleu':
                score = bleu_score(candidate, reference)
            elif mode == 'cider':
                score = cider_score(candidate, reference)
            elif mode == 'rouge':
                score = rouge_score(candidate, reference)
            elif mode == 'meteor':
                pass
            elif mode == 'spice':
                pass

            similarity_matrix[i, j] = score

    return np.mean(similarity_matrix)


def build_dataset(params):
    print('mode is {}'.format(params.mode))
    coco_ann_path = params.ann_path
    if not os.path.exists(params.output_dir):
        os.mkdir(params.output_dir)
    saved_file_path = os.path.join(params.output_dir, params.saved_mode_file_name + '_' + params.mode + '.npz')

    img_ids = []
    img_names = []
    captions = []
    similarity_scores = []

    if not os.path.exists(saved_file_path):
        assert os.path.exists(coco_ann_path)
        with open(coco_ann_path, 'r') as f:
            ann_file_info = json.load(f)

        #  图片id与图片captions的映射
        image_id_captions = {ann['image_id']: [] for ann in ann_file_info['annotations']}
        for ann in ann_file_info['annotations']:
            # 控制caption的个数
            if len(image_id_captions[ann['image_id']]) < 5:
                image_id_captions[ann['image_id']] += [ann['caption']]

        #  图片id与图片路径的映射
        images_id_name = {img['id']: img['file_name'] for img in ann_file_info['images']}

        # 计算图片id与相应相似度组成的字典
        image_id_similarity = {}
        for (img_id, caps) in image_id_captions.items():
            similarity_score = calculate_similarity(caps, mode=params.mode)
            image_id_similarity[img_id] = similarity_score
            print('image:{}, {} score:{}'.format(img_id, params.mode, similarity_score))

        # 从小到大排序, [(img_id, sim_value)]
        image_id_similarity_sorted = dict(sorted(image_id_similarity.items(), key=lambda x: x[1], reverse=False))

        # 构造数据集
        for item in image_id_similarity_sorted.items():
            img_id, score = item

            img_ids.append(img_id)
            img_names.append(images_id_name[img_id])
            captions.append(image_id_captions[img_id])
            similarity_scores.append(score)

        # 保存数据集
        np.savez(saved_file_path, image_id=img_ids, image_name=img_names, caption=captions, similarity_score=similarity_scores)

    else:
        saved_dataset = np.load(saved_file_path)
        img_ids = saved_dataset['image_id'].tolist()
        img_names = saved_dataset['image_name'].tolist()
        captions = saved_dataset['caption'].tolist()
        similarity_scores = saved_dataset['similarity_score'].tolist()
    return img_ids, img_names, captions, similarity_scores


def gen_labels(selected_img_names):
    lable_num = len(selected_img_names)
    one_label_num = int(lable_num / 2 + 0.5)
    zero_label_num = lable_num - one_label_num
    labels = np.ones(one_label_num, dtype=int).tolist() + np.zeros(zero_label_num, dtype=int).tolist()

    image_name_label = dict(zip(selected_img_names, labels))

    random.shuffle(selected_img_names)
    labels = [image_name_label[img_name] for img_name in selected_img_names]
    return labels


def main():
    # 设置参数
    params = Parameters()

    # dataset = build_dataset(params)
    # for img_name, captions in dataset:
    #     print(img_name, captions)
    img_ids, img_names, captions, similarity_scores = build_dataset(params)

    selected_img_names = img_names[:20000] + img_names[-20000:]
    labels = gen_labels(selected_img_names)

    train_img_names = selected_img_names[:30000]
    train_labels = labels[:30000]

    val_img_names = selected_img_names[30000:35000]
    val_labels = labels[30000:35000]

    test_img_names = selected_img_names[35000:40000]
    test_labels = labels[35000:40000]

    train_dataset = {
        'image_name': train_img_names,
        'label': train_labels
    }
    val_dataset = {
        'image_name': val_img_names,
        'label': val_labels
    }
    test_dataset = {
        'image_name': test_img_names,
        'label': test_labels
    }
    dataset = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    if not os.path.exists(params.output_dir):
        os.mkdir(params.output_dir)
    dataset_path = os.path.join(params.output_dir, params.mode_dataset_name + '_' + params.mode + '.json')

    with open(dataset_path, 'w') as f:
        json.dump(dataset, f)
    print('finished')


if __name__ == '__main__':
    print('coco dataset')
    main()
