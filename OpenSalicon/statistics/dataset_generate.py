import cv2
import json
import os

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.config import cfg
"""
这部分代码读取COOC和Salicon数据集的信息，生成一个文件夹IJCAI2020,
下面有images，fixation_maps和annotation三个子文件夹，
分别存放salicon中的图片，图片对应的fixation map，图片对应的captions
"""


def process(saved_dataset_dir, data_type):
    print('processing {} dataset'.format(data_type))
    image_dir = os.path.join(saved_dataset_dir, 'images', data_type)
    image_map_dir = os.path.join(saved_dataset_dir, 'fixation_maps', data_type)
    annotation_dir = os.path.join(saved_dataset_dir, 'annotation')

    # 读取图片名（只有前缀，没有类型）
    with open(cfg.DATASETS.SALICON.DATASET_FILE_PATH, encoding='utf-8') as file:
        dataset_files = json.loads(file.read())
    img_names = dataset_files[data_type]
    # 加载coco annotation文件
    if data_type == 'train':
        ann_file_path = cfg.DATASETS.COCO.TRAIN_ANNOTATIONS_FILE_PATH
    elif data_type == 'val':
        ann_file_path = cfg.DATASETS.COCO.VAL_ANNOTATIONS_FILE_PATH
    else:
        print('data_type error')
        return
    with open(ann_file_path, 'r') as f:
        ann_file_info = json.load(f)

    # coco image id 到 coco image name 的映射
    img_id_names = {image['id']: image['file_name'] for image in ann_file_info['images']}

    # coco image id 到 coco image captions
    img_id_captions = {ann['image_id']: [] for ann in ann_file_info['annotations']}
    for ann in ann_file_info['annotations']:
        img_id_captions[ann['image_id']] += [ann['caption']]

    # coco image name 到 coco image captions 的映射
    img_name_captions = {img_name: img_id_captions[img_id] for img_id, img_name in img_id_names.items()}

    salicon_img_name_captions = {}
    for img_name in img_names:
        image_path = cfg.DATASETS.SALICON.IMAGE_PATH % (data_type, img_name)
        map_path = cfg.DATASETS.SALICON.MAP_IMAGE_PATH % (data_type, img_name)
        image = cv2.imread(image_path)
        image_map = cv2.imread(map_path, 0)

        cv2.imwrite(os.path.join(image_dir, img_name + '.jpg'), image)
        cv2.imwrite(os.path.join(image_map_dir, img_name + '.png'), image_map)

        salicon_img_name_captions[img_name + '.jpg'] = img_name_captions[img_name + '.jpg']

    print('{} image and fixation map saved!'.format(data_type))

    with open(os.path.join(annotation_dir, data_type + '_salicon_captions.json'), 'w') as f:
        json.dump(salicon_img_name_captions, f)
    print('{} caption saved!'.format(data_type))


def main():
    saved_dataset_dir = '/Shared_Resources/social_media/DataSet/IJCAI_2020/'
    process('train', saved_dataset_dir)
    process('val', saved_dataset_dir)


def test():
    print(cfg.DATASETS.COCO.TRAIN_ANNOTATIONS_FILE_PATH)
    print(cfg.DATASETS.COCO.VAL_ANNOTATIONS_FILE_PATH)

    print(os.path.exists(cfg.DATASETS.COCO.TRAIN_ANNOTATIONS_FILE_PATH))
    print(os.path.exists(cfg.DATASETS.COCO.VAL_ANNOTATIONS_FILE_PATH))


if __name__ == "__main__":
    main()
