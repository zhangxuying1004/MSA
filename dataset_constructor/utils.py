class Parameters:
    def __init__(self):
        self.image_dir = '/home/zhangxuying/DataSet/COCO/train2014/'
        self.ann_path = '/home/zhangxuying/DataSet/COCO/annotations/captions_train2014.json'
        self.output_dir = '/home/zhangxuying/Project/Paper_code/MSA/dataset_constructor/output/'

        # diversity / consistent
        self.mode = 'cider'
        self.saved_mode_file_name = 'sorted_coco_dataset_mode'
        self.mode_dataset_name = 'dataset_mode'

        # topic
        self.topic_num = 10
        self.image_name_file_name = 'image_name.json'
        self.text_file_name = 'text.json'
        self.saved_topic_file_name = 'sorted_coco_dataset_topic'
