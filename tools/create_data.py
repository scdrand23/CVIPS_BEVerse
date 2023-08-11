# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
import icecream as ic
from tools.data_converter import carla_converter as carla_converter
from tools.data_converter.create_gt_database import create_groundtruth_database
import sys
import os

def carla_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """

    import os
    info_save_path = '/home/datasets2/CLCVQ/VruCoPNet/DeepAccident/DeepAccidentNpz/data/VRU_data/carla_infos'
    os.makedirs(info_save_path, exist_ok=True)
    if 'mini' in version:
        carla_converter.create_carla_infos_mini(root_path,
                                           info_prefix,
                                           version=version, info_save_path=info_save_path)
    else:
        # a = 5
        # ic(a)
        carla_converter.create_carla_infos(root_path,
                                           info_prefix,
                                           version=version, info_save_path=info_save_path)
        


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='carla', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/carla',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/carla',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='carla')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':    
    train_version = f'{args.version}'
    carla_data_prep(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        version=train_version,
        dataset_name='CarlaDataset',
        out_dir=args.out_dir)
