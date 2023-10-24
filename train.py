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
import wandb

from vit_pytorch.vivit import ViT


seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


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


if __name__=='__main__':


    wandb.init(project="attention-classification", name="exp-1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ... (previous code)


    root_dir    = './dataset/learning'
    train_dir   = os.path.join(root_dir, 'train')
    test_dir    = os.path.join(root_dir, 'test')


    train_dataset   = VideoDataset(train_dir, transform=data_transform)
    test_dataset    = VideoDataset(test_dir, transform=data_transform)

    batch_size      = 16

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader   = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)

    ####################################################################################################3

    model = ViT(
        image_size = (240,320),          # image size - height and width
        frames = 5,                    # number of frames
        image_patch_size = (80,80),      # image patch size
        frame_patch_size = 5,           # frame patch size
        num_classes = 2,
        dim = 64,
        spatial_depth = 4,               # depth of the spatial transformer
        temporal_depth = 4,              # depth of the temporal transformer
        heads = 4,
        mlp_dim = 128
    )




    ####################################################################################################

    learning_rate   = 3e-4
    num_epochs      = 100
    # Create model, loss function, and optimizer
    criterion       = nn.CrossEntropyLoss()
    # For classification tasks
    optimizer       = optim.Adam(model.parameters(), lr=learning_rate)



    # Move model and loss function to GPU
    model.to(device)
    criterion.to(device)

    checkpoint_dir  = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_interval = 10  # Save checkpoint every 20 epochs
    
    train_losses = []  # List to store training losses
    test_losses = []    # List to store validation losses
    
    start_time = time.time()

    for epoch in range(num_epochs):

        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_batch = 0
        total_batch = 0
        # Wrap the data loader with tqdm to create a progress bar
        train_loader_with_progress = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] (Training)')
        
        for videos, labels in train_loader_with_progress:
            

            videos = videos.to(device)  # Move input data to GPU
            labels = labels.to(device)

 
            # Forward pass
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(videos)
            loss = criterion(outputs, labels)
            #print(loss.item())
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

                # Calculate batch accuracy
            _, predicted_batch = outputs.max(1)
            total_batch += labels.size(0)
            correct_batch += predicted_batch.eq(labels).sum().item()

            # Update the tqdm progress bar for batches
            train_loader_with_progress.set_postfix({'Train Loss (Batch)': loss.item(), 'Train Acc (Batch)': 100. * correct_batch / total_batch, 'Train Loss': running_loss / len(train_loader)})
        
        # Calculate epoch accuracy
        epoch_accuracy = 100. * correct_batch / total_batch


        # Update the tqdm progress bar for the epoch
        train_loader_with_progress.set_postfix({'Train Loss': running_loss / len(train_loader), 'Train Acc': epoch_accuracy})
    

        


        # Validation
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            # Wrap the validation data loader with tqdm
            test_loader_with_progress = tqdm(test_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] (Test)')
            
            for videos, labels in test_loader_with_progress:
                videos = videos.to(device)  # Move input data to GPU
                labels = labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update the tqdm progress bar
                test_loader_with_progress.set_postfix({'Test Loss': test_loss / len(test_loader), 'Test Acc': 100. * correct / total})
        
        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.8f} - Test Acc: {100. * correct / total:.8f}%")
        wandb.log({"Training loss": avg_train_loss, "Test loss": avg_test_loss, "Test accuracy" :100. * correct / total })


       # Save checkpoint every checkpoint_interval epochs
        if epoch == 0 or ((epoch + 1) % checkpoint_interval == 0):
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'test_accuracy': 100. * correct / total
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    wandb.finish()

    end_time = time.time()
    total_training_time = end_time - start_time
    # Save the final list of training and validation losses
    losses_dict = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time':total_training_time
    }
    
    losses_path = os.path.join(checkpoint_dir, 'losses.pt')
    torch.save(losses_dict, losses_path)
    