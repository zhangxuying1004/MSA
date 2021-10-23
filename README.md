# MSA(Mode-Sensitive Attention)
This repository contains the reference code for our paper _Exploring Language Prior for Mode-Sensitive Visual Attention Modeling_. You can click [Paper link](https://github.com/zhangxuying1004/MSA/blob/master/files/ACM%20MM2020_Exploring%20Language%20Prior%20for%20Mode-Sensitive%20Visual%20Attention%20Modeling.pdf) to download our paper or click [Presentation link](https://dl.acm.org/doi/abs/10.1145/3394171.3414008) to watch the presentation we made in ACM MultiMedia 2020.  

## 1 Introduction

This repo is the implementation of "Exploring Language Prior for Mode-Sensitive Visual Attention Modeling".  
This repo consists of five folders as follows:  

>> - dataset_constructor
>> - mode_classifier
>> - topic_classifier
>> - OpenSalicon
>> - Benchmark
>> - Futrure_work

### 1.1 dataset_constructor

The dataset_constructor folder is used to build a dataset which is based on COCO train dataset and made up of multiple modes/topics.  
In this folder, the sub_folder caption_metrics provides the compute methods of image caption metrics which are used in coco_mode.py and coco_topic.py; the utils.py file defines the parameters in coco_mode.py and coco_topic.py; the coco.py is used to build diverse/consistent mode dataset according to the captions average CIDEr score, and the coco2.py is used to build multiple topics dataset according to caption text clustering.  

### 1.2 mode_classifier

The mode_classifier folder uses diverse/consistent mode dataset generated in mode_coco.py to train and test a classifier whose input is a image and output is the image mode(diverse or consistent).  

### 1.3 topic_classifier

The topic_classifier uses multiple topics dataset generated in topic_coco.py to train a classifier whose input is a image and output is the image topic id.  

### 1.4 OpenSalicon

The OpenSalicon folder is the core of this repo.  

- First, enter the statistics sub_folder, use mode_dataset_statistics.py to call the classifier trained in mode_classifier folder in order to divide the Salicon dataset into two modes sub_dataset, diverse salicon dataset and consistent salicon dataset; use topic_dataset_statistics.py to call the classifier trained in topic_classifier folder in order to divide the Salicon dataset into ten topics sub_dataset.  
- Second, the main task is to train OpenSalicon model whose input is a image and output is the salient region. The code is in salicon.py and core sub_folder. We use the diverse/consistent mode datasets to train two modes OpenSalicon models respectively and use ten topic datasets to train ten topic OpenSalicon models respectively.  
- Finally, the main task is to use the salicon models trained before to generate fixation map in three benchmark dataset, bruce, judMad and pascal. In test stage, the input image is processed by every trained OpenSalicon model, and then we combine the OpenSalicon results according to the output tensor of binary_classifier/topic_classifier. Besides, center bias and GaussianBlur are also used in the final result. The implementation details is shown in fixation_map sub_folder.  

### 1.5 Benchmark

The benchmark folder contains matlab codes to test the generated fixation maps in three benchmark dataset.  

### 1.6 Future work

The Futrure_work folder contains some expanding experiments.  

## 2 Requirements

```python
pip install -r requirements.txt
```

## 3 Run procedure

- Firstly, enter dataset_constructor dir and execute the operations described in the README.md to build dataset.  
- Secondly, enter mode_classifier folder and execute the operations described in the README.md to build mode classifier, and enter topic_classifier folder and execute the operations described in the README.md to build topic classifier.  
- Nextly, enter OpenSalicon folder and execute the operations described in the README.md to train open_salicon model and use the trained to generate fixation map in three benchmark dataset.  
- Finally, enter the Benchmark folder and execute the operations described in the README.md to test the fixation map generated by MSA model in five benchmark metrics.  

## 4 Expanding experiments

As is known to all, the fixation map label of a dataset is is hard to get. In this module, we try to use mattnet model to generate pseudo fixation maps for coco dataset. Then, we use the gained pseudo dataset to train opensalicon model. From the current experiment, we did not achieve the desired results. In future, We will explore further.
