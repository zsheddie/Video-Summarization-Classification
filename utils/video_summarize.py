import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

# 自定义视频数据集类
class VideoDataset(Dataset):
    def __init__(self, video_folder, feature_dim):
        self.video_folder = video_folder
        self.feature_dim = feature_dim
        self.video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.video_files[idx])
        frames = self._extract_frames(video_path)
        features = self._extract_features(frames)
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # 调整帧的大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)
    
    def _extract_features(self, frames):
        # 这里使用简单的平均值作为特征，实际应用中可以使用更复杂的特征提取方法，如预训练的CNN
        return frames.mean(axis=(1, 2, 3))  # 对每个帧取均值，得到一维特征

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)  # 可以加一层全连接层

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义参数
video_folder = 'E:/KIT/FZI_hiwi/Project/data/mixedUp'  # 视频文件夹路径
feature_dim = 1  # 特征维度，这里是1，因为我们使用平均值作为特征
hidden_size = 64
num_layers = 2
batch_size = 5
num_epochs = 10

# 初始化数据集和数据加载器
dataset = VideoDataset(video_folder, feature_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = LSTMModel(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers)
model.eval()

# 提取每个视频的特征向量
video_summaries = []
with torch.no_grad():
    for inputs in dataloader:
        summaries = model(inputs)
        video_summaries.extend(summaries.numpy())

video_summaries = np.array(video_summaries)

# 执行OPTICS聚类
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics.fit(video_summaries)

# 获取聚类标签
labels = optics.labels_

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(video_summaries[:, 0], video_summaries[:, 1], c=labels, cmap='viridis')
plt.title('OPTICS Clustering')
plt.show()

# 输出聚类结果
unique_labels = np.unique(labels)
print(f"发现的簇数: {len(unique_labels)}")
