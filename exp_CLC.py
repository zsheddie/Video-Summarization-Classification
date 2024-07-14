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
from models.CLC import CNN_LSTM
from train.train_CLC import train_step, test_step
import math



label_dict = {}
all_label = []
train_label = []
test_label = []

    
class VideoDataset(Dataset):
    def __init__(self, video_folder, max_frames=241, resize_shape=(224, 224)):
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
        global train_label
        global test_label
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
                    elif folder_name == 'arm wrestling':
                        all_label.append(6)
                    elif folder_name == 'yoga':
                        all_label.append(7)
                    elif folder_name == 'feeding goats':
                        all_label.append(8)
                    elif folder_name == 'making a sandwich':
                        all_label.append(9)

        print(all_label)
        self.label = all_label
        if 'train' in video_folder:
            train_label = all_label
            all_label = []
        elif 'test' in video_folder:
            test_label = all_label
            all_label = []
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
        #print('$$$$$$$$$$$$$$$$$$$$', frames)
        #print('$$$$$$$$$$$$$$$$$$$$', self.label)
        #print(torch.tensor(frames, dtype=torch.float32).shape)
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(self.label[idx], dtype=torch.float32)
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
        frames_use = 9
        interval = math.floor(self.max_frames/(frames_use-1))
        #print(list(range(0, (self.max_frames+1), interval)))
        frames = [frames[i] for i in list(range(0, (self.max_frames+1), interval))]
        return torch.stack(frames).to(self.device)
    

###################################           Training               ####################


# 超参数
train_folder = 'E:/KIT/FZI_hiwi/Project/data/10choosen/train'  # 视频文件夹路径
test_folder = 'E:/KIT/FZI_hiwi/Project/data/10choosen/test'
cnn_out_dim = 512
lstm_hidden_dim = 256
num_layers = 2
n_clusters = 6
learning_rate = 0.001
num_epochs = 30
batch_size = 50
n_classes = 10

train_dataset = VideoDataset(train_folder) 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataset = VideoDataset(test_folder)  
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失函数和优化器
if torch.cuda.is_available():
    print('cuda is available!!!!!!!!!!!!!!!!!!!!!')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = CLK(cnn_out_dim, lstm_hidden_dim, num_layers, n_clusters).to(device)
model = CNN_LSTM(cnn_out_dim, lstm_hidden_dim, num_layers, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

all_label = torch.tensor(all_label).to(device)
train_step(model, train_dataloader, num_epochs, optimizer, device)

torch.save(model.state_dict(), 'E:/KIT/FZI_hiwi/Project/saved_models/CLC_006.pth')


# 提取每个视频的特征向量
video_summaries = []
with torch.no_grad():   
    out = test_step(model, test_dataloader, num_epochs, optimizer, device)

pred = np.argmax(out, axis=1)
print(pred)

test = np.array(test_label)
print(test)

#video_summaries = np.array(video_summaries)
np.savez_compressed('./saved_summaries/CLC_006.npz', arr1 = pred, arr2 = test)

acc = 0
for id in range(test.shape[0]):
    if pred[id] == test[id]:
        acc += 1

print(f'The final test accuracy is {(acc*100/(test.shape[0]))}%')