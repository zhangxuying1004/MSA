"""
注意点：使用h5py.File读取文件，关于键名，可以使用f.keys()查询，但是由于python的版本问题，可能查询不到，
解决这个问题的方式是，进入文件所在的目录，打开matlab，使用命令 load 文件名，加载文件，然后使用命令who，
即可看到文件中包含的变量名，也就是h5py.File读取文件时的键名。
"""
import os
import h5py
import numpy as np

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from OpenSalicon.utils.config import params


def switch_file_mat_to_np(params):
    mat_file_path = params.pascal_fixation_map_mat_file
    np_file_path = params.pascal_fixation_map_path
    assert os.path.exists(mat_file_path)

    f = h5py.File(mat_file_path)
    fixation_map_hdf5_data = f['fixmapCell']
    fixation_map_np_data = np.asarray([f[item[0]][:] for item in fixation_map_hdf5_data])

    np.savez(np_file_path, fixation_map=fixation_map_np_data)
    print('switch finished')


def main():
    pascal_fixation_map_mat_file = params.pascal_fixation_map_mat_file
    print(pascal_fixation_map_mat_file)
    switch_file_mat_to_np(params)


if __name__ == "__main__":
    main()
