import os
import json
if __name__ == "__main__":
    topic_num = 5
    dataset_path = os.path.join('/home/zhangxuying/Project/Paper_code/MSA/dataset_constructor/output/', 'dataset_' + str(topic_num) + '.json')
    print(os.path.exists(dataset_path))
    with open(dataset_path, 'r') as f:
        dataset_file = json.load(f)
    print(type(dataset_file))
    print(dataset_file.keys())
