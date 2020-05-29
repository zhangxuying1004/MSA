# Intoduction
## Run procedure
- First, enter the statistics sub_folder, use mode_dataset_statistics.py to call the classifier trained in mode_classifier folder in order 
to divide the Salicon dataset into two modes sub_dataset, diverse salicon dataset and consistent salicon dataset; use 
topic_dataset_statistics.py to call the classifier trained in topic_classifier folder in order to divide the Salicon dataset into ten 
topics sub_dataset.  
```linux
.../MSA/OpenSalicon/$ python mode_dataset_statistics.py
.../MSA/OpenSalicon/$ python topic_dataset_statistics.py
```
- Second, the main task is to train OpenSalicon model whose input is a image and output is the salient region. The code is in salicon.py 
and core sub_folder. We use the diverse/consistent mode datasets to train two modes OpenSalicon models respectively and use ten topic 
datasets to train ten topic OpenSalicon models respectively.  
```linux
.../MSA/OpenSalicon/$ python runner.py --gpu=0,1 --batch-size=23
```
record the trained model checkpoint path.
- Finally, the main task is to use the salicon models trained before to generate fixation map in three benchmark dataset, bruce, 
judd and pascal. In test stage, the input image is processed by every trained OpenSalicon model, and then we combine the OpenSalicon 
results according to the output tensor of binary_classifier/topic_classifier. Besides, center bias and GaussianBlur are also used in the 
final result. The implementation details is shown in fixation_map sub_folder. 
```linux
.../MSA/OpenSalicon/utils$ CUDA_VISIBLE_DEVICES=0,1 python gen_fixation_map.py
.../MSA/OpenSalicon/utils$ CUDA_VISIBLE_DEVICES=0,1 python gen_mode_combine_fixation_map.py
.../MSA/OpenSalicon/utils$ CUDA_VISIBLE_DEVICES=0,1 python gen_topic_combine_fixation_map.py
```
