

"""
Training script for Vivit

__author__ = 'Amit Parag'
__organization = 'Sintef Ocean AS'

"""


from __future__ import print_function

import glob
import os
import random
import shutil
import time
import av
import imageio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from tqdm import tqdm




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True




def make_datasets(root_dir = './dataset/learning', isValidation=False):

    
    
    source_slip_videos = root_dir + '/slip'  
    source_wriggle_videos = root_dir + '/wriggle' 
    
    classes = ['slip', 'wriggle']

    if isValidation:
        subsets = ['validation']
        split_ratios = [1.]  

    else:
        subsets = ['train', 'test']
        split_ratios = [0.9, 0.1] 

    
    # Create root directory
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Create subsets directories and copy videos
    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)
    
        for class_name in classes:
            class_dir = os.path.join(subset_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            source_videos = source_slip_videos if class_name == 'slip' else source_wriggle_videos
            video_files = [f for f in os.listdir(source_videos) if f.endswith('.avi')]
            random.shuffle(video_files)  # Shuffle video files
            
            split_ratio = split_ratios[subsets.index(subset)]
            num_videos = int(len(video_files) * split_ratio)
            
            for video_file in video_files[:num_videos]:
                src_path = os.path.join(source_videos, video_file)
                dest_path = os.path.join(class_dir, video_file)
                shutil.copy(src_path, dest_path)
    
    print("Directory structure and video copying completed.")






if __name__=='__main__':

    seed = 76
    seed_everything(seed)
    make_datasets(root_dir = './dataset/learning', isValidation=False)
