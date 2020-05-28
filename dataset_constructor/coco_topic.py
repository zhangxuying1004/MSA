import json
import os
from time import time
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from utils import Parameters


def captions_process(captions):
    captions_text = ''
    flag = True     # 判断是不是第一个caption
    for caption in captions:
        if flag:
            captions_text += caption
            flag = False
        else:
            captions_text = captions_text + ' ' + caption
    return captions_text


def preprocess(params):
    image_name_file_path = os.path.join(params.output_dir, params.image_name_file_name)
    text_file_path = os.path.join(params.output_dir, params.text_file_name)
    image_name, text = [], []
    # 如果文件已经生成，直接读取，如果文件没有生成，则生成并保存文件
    if os.path.exists(image_name_file_path) and os.path.exists(text_file_path):
        with open(image_name_file_path, 'r') as f:
            image_name = json.load(f)
        with open(text_file_path, 'r') as f:
            text = json.load(f)
    else:
        with open(params.ann_path, 'r') as f:
            coco_info = json.load(f)
        # image id to image captions
        image_id_captions = {ann['image_id']: [] for ann in coco_info['annotations']}
        for ann in coco_info['annotations']:
            image_id_captions[ann['image_id']] += [ann['caption']]
        # image id to image name
        image_id_name = {img['id']: img['file_name'] for img in coco_info['images']}

        for img_id, img_name in image_id_name.items():
            captions = image_id_captions[img_id]
            captions_text = captions_process(captions)
            image_name.append(img_name)
            text.append(captions_text.strip())
        with open(image_name_file_path, 'w') as f:
            json.dump(image_name, f)
        with open(text_file_path, 'w') as f:
            json.dump(text, f)
    return image_name, text


def text_vectorize(text, params):
    text_vector = None
    text_vector_file_path = os.path.join(params.output_dir, 'text_vector.npz')
    if not os.path.exists(text_vector_file_path):
        vectorizer = CountVectorizer(stop_words='english')
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(text))
        text_vector = tfidf.toarray()
        np.savez(text_vector_file_path, text_vector=text_vector)
    else:
        f = np.load(text_vector_file_path)
        text_vector = f['text_vector']
    return text_vector


def km_clustering(text_vector, params):
    clusters = None
    cluster_file_path = os.path.join(params.output_dir, params.saved_topic_file_name + '_' + str(params.topic_num) + '.json')
    if not os.path.exists(cluster_file_path):
        print('no')
        km = KMeans(n_clusters=params.topic_num)
        km.fit(text_vector)
        # 每张图片所属的label构成的列表
        clusters = km.labels_.tolist()

        with open(cluster_file_path, 'w') as f:
            json.dump(clusters, f)
    else:
        print('yes')
        with open(cluster_file_path, 'r') as f:
            clusters = json.load(f)
    return clusters


def test():
    params = Parameters()
    # 提取captions_train2014中的信息，得到图片名与对应的文本描述
    image_name, text = preprocess(params)
    print(len(image_name))
    print(len(text))

    time1 = time()
    text_vector = text_vectorize(text, params)
    print(text_vector)
    print(text_vector.shape)
    time2 = time()
    print('text vectorize costs {} s'.format(time2 - time1))


def main():
    params = Parameters()
    # 提取captions_train2014中的信息，得到图片名与对应的文本描述
    image_name, text = preprocess(params)
    print(len(image_name))
    print(len(text))

    time1 = time()
    text_vector = text_vectorize(text, params)
    print(text_vector)
    print(text_vector.shape)
    time2 = time()
    print('text vectorize costs {} s'.format(time2 - time1))

    time3 = time()
    cluster = km_clustering(text_vector, params)
    time4 = time()
    print('kmeans cluster costs {} s'.format(time4 - time3))
    # print(cluster)
    print(len(cluster))

    image_name_label = dict(zip(image_name, cluster))
    random.shuffle(image_name)
    label = [image_name_label[img_name] for img_name in image_name]

    train_image_name, train_label = image_name[0: 50000], label[0: 50000]
    val_image_name, val_label = image_name[50000: 70000], label[50000: 70000]
    test_image_name, test_label = image_name[70000:], label[70000:]

    train_data = {
        'image_name': train_image_name,
        'label': train_label
    }

    val_data = {
        'image_name': val_image_name,
        'label': val_label
    }
    test_data = {
        'image_name': test_image_name,
        'label': test_label
    }

    dataset = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    dataset_path = os.path.join(params.output_dir, 'dataset_topic_' + str(params.topic_num) + '.json')
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f)
    print('finished!')


def image_name_lable_mapping():
    params = Parameters()
    cluster_file_path = os.path.join(params.output_dir, params.saved_topic_file_name + '_' + str(params.topic_num) + '.json')
    with open(cluster_file_path, 'r') as f:
        cluster = json.load(f)
    print(len(cluster))

    image_name, _ = preprocess(params)
    print(len(image_name))

    dataset = dict(zip(image_name, cluster))
    with open(os.path.join('gen_data/', 'mapping_dataset_5.json'), 'w') as f:
        json.dump(dataset, f)
    print('finished!')


if __name__ == "__main__":
    # test()
    main()
    # image_name_lable_mapping()
