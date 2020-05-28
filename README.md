# MSA
## Introduction  
This repo is the implementation of "Exploring Language Prior for Mode-Sensitive Visual Attention Modeling". 
This repo consists of five folders as follows:
```
* dataset_constructor
* binary_classifation
* classifation
* OpenSalicon 
* Benchmark 
* Futrure_work
```
The dataset_constructor folder is used to build a dataset which is based on COCO train dataset and made up of multiple modes/topics.   
In this folder, the sub_folder caption_metrics provides the compute methods of image caption metrics which are used in coco_mode.py and coco_topic.py;  
the utils.py file defines the parameters in coco_mode.py and coco_topic.py;  
the coco.py is used to build diverse/consistent mode dataset according to the captions average CIDEr score, and the coco2.py is used to build multiple topics dataset according to caption text clustering.  
The mode_classifier folder uses diverse/consistent mode dataset generated in mode_coco.py to train and test a classifier whose input is a image and output is the image mode(diverse or consistent).   
The classifation folder s multiple topics dataset generated in topic_coco.py to train a classifier whose input is a image and output is the image topic id.     
The OpenSalicon folder is the core of this repo.   
First, enter the statistics sub_folder, use mode_dataset_statistics.py to call the classifier trained in mode_classifier folder in order to divide the Salicon dataset into two modes sub_dataset, diverse salicon dataset and consistent salicon dataset; use topic_dataset_statistics.py to call the classifier trained in topic_classifier folder in order to divide the Salicon dataset into ten topics sub_dataset.   
Second, the main task is to train OpenSalicon model whose input is a image and output is the salient region. The code is in salicon.py and core sub_folder. We use the diverse/consistent mode datasets to train two modes OpenSalicon models respectively and use ten topic datasets to train ten topic OpenSalicon models respectively.   
Finally, the main task is to use the salicon models trained before to generate fixation map in three benchmark dataset, bruce, judd and pascal. In test stage, the input image is processed by every trained OpenSalicon model, and then we combine the OpenSalicon results according to the output tensor of binary_classifier/topic_classifier. Besides, center bias and GaussianBlur are also used in the final result. The implementation details is shown in fixation_map sub_folder.   
The benchmark folder contains matlab codes to test the generated fixation maps in three benchmark dataset.  
The Futrure_work folder contains some expanding experiments

## requirements
```
pip install -r requirements.txt
```
## run procedure
Firstly, enter dataset_constructor dir and execute the operations described in the README.md to build dataset.
Secondly, enter mode_classifier folder and execute the operations described in the README.md to build mode classifier, and enter topic_classifier folder and execute the operations described in the README.md to build topic classifier, 
Nextly, enter OpenSalicon folder and execute the operations described in the README.md to train open_salicon model and use the trained to generate fixation map in three benchmark dataset.
Finally, enter the OpenSalicon dir and execute the operations described in the README.md to train and test MSA model.

## Expanding experiments
As is known to all, the fixation map label is is hard to get. In this module, we try to use mattnet model to generate pseudo fixation maps for coco dataset. Then we use the gained pseudo dataset to train opensalicon model. From the current experiment, we did not achieve the desired results. In future, We will explore further. 
 
