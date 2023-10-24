import os
import random
import glob
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
import imageio
from vit_pytorch.vivit import ViT
torch.device('cpu')



def make_validation_datasets():

    
    root_dir = './dataset/unseen_validation_data/'
    source_slip_videos = './dataset/unseen_validation_data/slip'  
    source_wriggle_videos = './dataset/unseen_validation_data/wriggle' 
    
    classes = ['slip', 'wriggle']
    subsets = ['validation']
    split_ratios = [1.0]  # Train, validation, test split ratios
    
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


class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.videos = self._load_videos()
        self.transform = transform

    def _load_videos(self):
        videos = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for video_file in os.listdir(class_dir):
                if video_file.endswith('.avi'):
                    video_path = os.path.join(class_dir, video_file)
                    videos.append((video_path, self.class_to_idx[class_name]))
        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]

        video = imageio.get_reader(video_path, 'ffmpeg')  # Open video with imageio
        
        #print("Video:", video_path)
        #print("Number of frames:", len(video))
        #print("Height:", video.get_meta_data()["size"][1])
        #print("Width:", video.get_meta_data()["size"][0])
        
        
        frames = []
        
        for frame in video:
            frame = frame[:, :, :3]  # Keep only the first three channels (RGB)
            frames.append(frame)

        video.close()
        
        #container = av.open(video_path)  # Open the video file with pyav
        #frames = []
        #for frame in container.decode(video=0):  # Loop through video frames
        #    img = frame.to_image()
        #    img = img.convert('RGB')  # Convert to RGB format
        #    frame_array = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], 3)
        #    frames.append(frame_array.numpy())
        
        #container.close() 
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            video_tensor = torch.stack(frames)
        #print(video_tensor.shape)
        return video_tensor.permute(1, 0, 2, 3), label  # Permute to (batch, channels, frames, height, width)

data_transform = Compose([
    ToTensor(),
])