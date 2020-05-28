import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.config import cfg, params


if __name__ == '__main__':
    print(cfg.DATASETS.SALICON.FIXATION_MAP_PATH)
    print(params.pascal_fixation_map_path)
