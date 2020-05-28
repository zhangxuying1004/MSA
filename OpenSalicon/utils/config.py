from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Datasets
__C.DATASETS = edict()
__C.DATASETS.SALICON = edict()

__C.DATASETS.SALICON.DATASET_FILE_PATH = '/home/luoyunpeng/DeepSaliencyED/datasets/Salicon.json'
__C.DATASETS.SALICON.IMAGE_PATH = '/home/luoyunpeng/dataset/salicon/images/%s/%s.jpg'
# __C.DATASETS.SALICON.FIXATION_MAP_PATH = '/home/luoyunpeng/dataset/salicon/fixations/%s/%s.mat'
__C.DATASETS.SALICON.FIXATION_MAP_PATH = '/Shared_Resources/social_media/DataSet/salicon/maps/%s/%s.png'
__C.DATASETS.SALICON.MAP_IMAGE_PATH = '/Shared_Resources/social_media/DataSet/salicon/maps/%s/%s.png'
__C.DATASETS.SALICON.CENTER_BIAS = '/home/luoyunpeng/dataset/salicon/centerbias.npy'

__C.DATASETS.TEST = edict()
__C.DATASETS.TEST.INAGE_FOLDER = '/home/luoyunpeng/dataset/salicon/images/%s'
__C.DATASETS.TEST.CENTER_BIAS = '/home/luoyunpeng/dataset/salicon/centerbias.npy'

# COCO
__C.DATASETS.COCO = edict()
__C.DATASETS.COCO.TRAIN_ANNOTATIONS_FILE_PATH = '/Shared_Resources/social_media/DataSet/COCO/annotations/captions_train2014.json'
__C.DATASETS.COCO.VAL_ANNOTATIONS_FILE_PATH = '/Shared_Resources/social_media/DataSet/COCO/annotations/captions_val2014.json'


# Dataset
__C.DATASET = edict()
__C.DATASET.DATASET_NAME = 'SALICON'
__C.DATASET.CENTER_BIAS = None
__C.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.DATASET.STD = [0.229, 0.224, 0.225]


# Common
__C.CONST = edict()
__C.CONST.DEVICE = '0'
__C.CONST.NUM_WORKER = 4
__C.CONST.RNG_SEED = 0
__C.CONST.BATCH_SIZE = 64
__C.CONST.WEIGHTS = None
# __C.CONST.WEIGHTS = './history_info/mse_output/checkpoints/2019-11-09T23-01-07.943920/ckpt-epoch-0010.pth.tar'
__C.CONST.IMG_W = 224  # 224, 448, 336
__C.CONST.IMG_H = 224  # 224, 448, 336
__C.CONST.IMG_C = 3
__C.CONST.CROP_IMG_W = 460  # 460, 580,
__C.CONST.CROP_IMG_H = 620  # 620, 780
__C.CONST.CROP_IMG_C = 3

# Directories
__C.DIR = edict()

# __C.DIR.OUT_PATH = './test'
# __C.DIR.OUT_PATH = './ijcai_output'
# __C.DIR.OUT_PATH = './topic_output'
__C.DIR.OUT_PATH = './pseudo_coco_output'

# Network
__C.NETWORK = edict()
__C.NETWORK.EPS = 1e-24
__C.NETWORK.LOSS_POS_WEIGHT = 15
__C.NETWORK.DOG_KERNEL_SIZES = []

# Network
__C.NETWORK = edict()
__C.NETWORK.EPS = 1e-24
__C.NETWORK.LOSS_POS_WEIGHT = 15
__C.NETWORK.DOG_KERNEL_SIZES = []

# Train
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.NUM_EPOCHES = 10
__C.TRAIN.SAVE_FREQ = 1
__C.TRAIN.GAUSSIAN_NOISE_MEAN = 0
__C.TRAIN.GAUSSIAN_NOISE_STD = 0.01
__C.TRAIN.READOUT_LEARNING_RATE = 1e-5
__C.TRAIN.READOUT_LR_MILESTONES = [50]
__C.TRAIN.GAMMA = .2
__C.TRAIN.BETAS = (0.9, 0.999)
__C.TRAIN.WEIGHT_DECAY = 5e-5
__C.TRAIN.SAVE_FREQ = 10
__C.TRAIN.use_center_bias = True

# 如果要在某个模型的基础上训练，修改__C.CONST.WEIGHTS为base模型的路径，修改__C.TRAIN.RESUME_TRAIN为True


class Parameters:
    def __init__(self):
        # dataset*
        self.mode_salicon_dataset_path = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/work/IJCAI/data/salicon_statistics.json'     # diverse + consistent
        self.topic_salicon_dataset_dir = '/Shared_Resources/social_media/project/zxy/PAMI/OpenSalicon/work/IJCAI/data/'

        self.coco_image_dir = '/Shared_Resources/social_media/DataSet/COCO/train2014/'
        self.pseudo_coco_file_path = '/Shared_Resources/social_media/DataSet/COCO/MM/pseudo_coco.json'

        # mattnet
        self.pseudo_mattnet_coco_fixation_map_dir = '/Shared_Resources/social_media/DataSet/COCO/MM/map/'
        self.mattnet_special_images = ['COCO_train2014_000000004308']
        # aws
        self.pseudo_aws_coco_fixation_map_dir = '/Shared_Resources/social_media/DataSet/COCO/MM/AWS/map/'
        self.aws_special_images_dir = '/Shared_Resources/social_media/DataSet/COCO/MM/aws_special_images.json'

        # pascal
        self.pascal_image_dir = '/Shared_Resources/data/XSSUN/datasets/imgs/pascal/'
        self.pascal_fixation_map_mat_file = '/home/data/XSSUN/datasets/fixations/pascalFixmap.mat'
        self.pascal_image_name_path = '/Shared_Resources/data/XSSUN/datasets/pascal_image_name.json'
        self.pascal_fixation_map_path = '/Shared_Resources/data/XSSUN/datasets/pascal_fixation_map.npz'


params = Parameters()
