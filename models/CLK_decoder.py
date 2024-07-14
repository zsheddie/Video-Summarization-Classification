## CNN_LSTM_K-means

'''
重构损失 (Reconstruction Loss)：用于训练自编码器，使得输入数据在通过编码和解码后能够尽可能重构原始输入。
KL 散度损失 (KL Divergence Loss)：用于度量当前的软分配和目标分配之间的差异。
优点：

自动聚类数:DEC 不需要预先指定聚类的数量，因为它使用目标分布来指导聚类过程。
柔性聚类:使用软分配(soft assignment),在处理边界点时更具弹性。
良好的特征学习:通过联合优化自编码器和聚类目标,DEC 可以学习到更好的特征表示。
缺点：

复杂度:DEC 的训练过程相对复杂，需要进行多次目标分布的更新和优化。
对初始聚类敏感: DEC 依赖初始的 K-means 聚类结果，初始聚类的好坏会影响最终结果。

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim as optim
from torchvision import models, transforms
from sklearn.cluster import KMeans
import numpy as np

class CNN_LSTM(nn.Module):
    def __init__(self, cnn_out_dim, lstm_hidden_dim, num_layers):
        super(CNN_LSTM, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(cnn_out_dim, lstm_hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, cnn_out_dim),
            nn.ReLU(),
            nn.Linear(cnn_out_dim, 112 * 112 * 3 * 5),
            nn.Sigmoid()
        )
    

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        cnn_out = []
        for i in range(seq_len):
            with torch.no_grad():
                #print(x.shape)
                cnn_out.append(self.cnn(x[:, i, :, :, :]).view(batch_size, -1))  ## （batch, frames, channel, height, weight）
        cnn_out = torch.stack(cnn_out, dim=1)
        #print(cnn_out.shape)    ## (batch size, frames, 512)
        lstm_out, _ = self.lstm(cnn_out)
        hidden = lstm_out[:, -1, :]

        reconstructed_frames = self.decoder(hidden).view(batch_size, seq_len, 3, 112, 112)
        return reconstructed_frames, hidden
    
class CLK(nn.Module):
    def __init__(self, cnn_out_dim, lstm_hidden_dim, num_layers, n_clusters):
        super(CLK, self).__init__()
        self.cnn_lstm = CNN_LSTM(cnn_out_dim, lstm_hidden_dim, num_layers)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, lstm_hidden_dim))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = n_clusters

    def initialize_cluster_centers(self, dataloader):
        features_list = []
        for videos in dataloader:
            videos = videos.to(self.device)
            with torch.no_grad():
                features = self.cnn_lstm(videos)
            features_list.append(features.cpu().numpy())
        all_features = np.vstack(features_list)
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++').fit(all_features)
        self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

    def forward(self, x):
        features = self.cnn_lstm(x)
        return features

    def compute_kmeans_loss(self, features):
        distances = torch.cdist(features, self.cluster_centers)
        min_distances, cluster_assignments = torch.min(distances, dim=1)
        return min_distances.mean(), cluster_assignments
    

