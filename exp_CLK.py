import os
import sys
sys.path.append('.')
sys.path.append('E:/KIT/FZI_hiwi/Project')

import cv2
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.CLK import CNN_LSTM, CLK
from train.train_CLK import train_step, test_step



label_dict = {}
all_label = []

# 自定义视频数据集类
class RES_VideoDataset(Dataset):
    def __init__(self, video_folder, max_frames=50, resize_shape=(224, 224)):
        self.video_folder = video_folder
        self.video_files2 = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        self.max_frames = max_frames
        self.resize_shape = resize_shape
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet18(pretrained=True).to(self.device)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))  # 去掉最后一层全连接层
        self.resnet.eval()
        all_video = []
        for subdir in os.listdir(video_folder):
            print(subdir)
            subdir = video_folder  + '/' + subdir
            for file in os.listdir(subdir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    folder_name = os.path.basename(subdir)
                    label_dict[file] = folder_name
                    all_video.append(file)
        print(all_video)
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(self.video_files2)
        self.video_files = all_video

    def __len__(self):
        return len(self.video_files)

    def traverse_directory(root_dir):
        all_video = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    all_video.append(file)
        return all_video
        
    def __getitem__(self, idx):
        #print(self.video_files[idx][:-4])
        video_path = os.path.join(self.video_folder, label_dict[self.video_files[idx]], self.video_files[idx])
        frames = self._extract_frames(video_path)
        features = self._extract_features(frames)
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        # 如果帧数不足max_frames，则重复最后一帧
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        return torch.stack(frames).to(self.device)
    
    def _extract_features(self, frames):
        with torch.no_grad():
            features = self.resnet(frames).squeeze(-1).squeeze(-1)
        return features.cpu().numpy()  # shape: (max_frames, 512)
    
    
class VideoDataset(Dataset):
    def __init__(self, video_folder, max_frames=200, resize_shape=(224, 224)):
        self.video_folder = video_folder
        self.max_frames = max_frames
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        all_video = []
        global all_label
        for subdir in os.listdir(video_folder):
            print(subdir)
            subdir = video_folder  + '/' + subdir
            for file in os.listdir(subdir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    folder_name = os.path.basename(subdir)
                    label_dict[file] = folder_name
                    all_video.append(file)
                    if folder_name == 'drinking':
                        all_label.append(0)
                    elif folder_name == 'driving car':
                        all_label.append(1)
                    elif folder_name == 'playing accordion':
                        all_label.append(2)
                    elif folder_name == 'rock climbing':
                        all_label.append(3)
                    elif folder_name == 'skiing slalom':
                        all_label.append(4)
                    elif folder_name == 'texting':
                        all_label.append(5)
                    
        print(all_video)
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        self.video_files = all_video
        print(self.video_files)
        

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        #print(self.video_files[idx][:-4])
        video_path = os.path.join(self.video_folder, label_dict[self.video_files[idx]], self.video_files[idx])
        
        frames = self._extract_frames(video_path)   #.unsqueeze(-1)
        '''
        if label_dict[self.video_files[idx]] == 'drinking':
            frames[-1] = 0
        elif label_dict[self.video_files[idx]] == 'driving car':
            frames[-1] = 1
        elif label_dict[self.video_files[idx]] == 'playing accordion':
            frames[-1] = 2
        elif label_dict[self.video_files[idx]] == 'rock climbing':
            frames[-1] = 3
        elif label_dict[self.video_files[idx]] == 'skiing slalom':
            frames[-1] = 4
        elif label_dict[self.video_files[idx]] == 'texting':
            frames[-1] = 5
        '''
        #print(torch.tensor(frames, dtype=torch.float32).shape)
        return torch.tensor(frames, dtype=torch.float32)
        #return frames.clone().detach()
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames_tensor = torch.stack(frames)
        return frames_tensor

    def extract_frames(self, video_path):
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV读取的是BGR格式，转换为RGB
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        # 如果帧数不足max_frames，则重复最后一帧
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        #print(torch.stack(frames).shape)
        return torch.stack(frames).to(self.device)
    

###################################           Training               ####################

# 超参数
video_folder = 'E:/KIT/FZI_hiwi/Project/data/6choosen'  # 视频文件夹路径
cnn_out_dim = 512
lstm_hidden_dim = 256
num_layers = 2
n_clusters = 6
learning_rate = 0.01
num_epochs = 5
batch_size = 8

dataset = VideoDataset(video_folder)   ## only an instance
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLK(cnn_out_dim, lstm_hidden_dim, num_layers, n_clusters).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_step(model, dataloader, num_epochs, optimizer, device)



# 提取每个视频的特征向量
video_summaries = []
with torch.no_grad():
    pred = test_step(model, dataloader, num_epochs, optimizer, device)

pred = np.array(pred)
#video_summaries = np.array(video_summaries)
np.savez_compressed('./saved_summaries/CLK_001.npz', arr1 = pred)
