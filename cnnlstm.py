import os
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

label_dict = {}

# 自定义视频数据集类
class VideoDataset(Dataset):
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
        #print(self.video_files)
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



##################################             CNN-LSTM                #############################
# 定义CNN-LSTM模型
class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_feature_size, lstm_hidden_size, lstm_num_layers):
        super(CNNLSTMModel, self).__init__()
        self.lstm = nn.LSTM(cnn_feature_size, lstm_hidden_size[0], lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size[0], lstm_hidden_size[1])  # 可以加一层全连接层
        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0)
        )

    def forward(self, x):
        # x = self.cnn(x)
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # 取最后一个时间步的输出
        return out


###################################           Training               ####################


# 定义参数
video_folder = 'E:/KIT/FZI_hiwi/Project/data/6choosen'  # 视频文件夹路径
cnn_feature_dim = 128  # 使用ResNet提取的特征维度
lstm_hidden_size = [64, 128]
lstm_num_layers = 2
batch_size = 128
num_epochs = 50
learning_rate = 0.001

# 初始化数据集和数据加载器
dataset = VideoDataset(video_folder)   ## only an instance
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMModel(cnn_feature_size=cnn_feature_dim, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers).to(device)
criterion = nn.MSELoss()  # 可以根据任务选择不同的损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
for epoch in range(num_epochs):
    if epoch > 30:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for inputs in tqdm(dataloader):
        inputs = inputs.to(device)
        print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        print(outputs.shape)
        # 假设目标是与输入特征相同，可以修改为具体任务
        loss = criterion(outputs, inputs[:, -1, :])
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


#####################################           Clustering                 ######################


# 切换模型到评估模式
model.eval()

# 提取每个视频的特征向量
video_summaries = []
with torch.no_grad():
    for inputs in dataloader:
        inputs = inputs.to(device)
        summaries = model(inputs)
        video_summaries.extend(summaries.cpu().numpy())

video_summaries = np.array(video_summaries)
np.savez_compressed('./saved_summaries/002.npz', arr1 = video_summaries)

# 执行OPTICS聚类
## min_samples: 核心对象的最小样本数，较小的min_samples可以找到更小的聚类
## xi: 相邻样本之间的最小距离，xi值越小，不同聚类之间的距离越小，产生的聚类数目会增加
## min_cluster_size：规定了一个聚类最少包含的样本数目
optics = OPTICS(min_samples=4, xi=0.02, min_cluster_size=3) 
optics.fit(video_summaries)

# 获取聚类标签
labels = optics.labels_

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(video_summaries[:, 0], video_summaries[:, 1], c=labels, cmap='viridis')
plt.title('OPTICS Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 输出聚类结果
unique_labels = np.unique(labels)
print(f"Clusters found: {len(unique_labels)}")
