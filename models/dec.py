## Deep Embedded Clustering

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

class RES_LSTM(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_layers):
        super(RES_LSTM, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去掉最后的全连接层
        self.lstm = nn.LSTM(cnn_output_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        return r_out[:, -1, :]  # 取最后一个时间步的输出作为特征
    
class CNN_LSTM(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_layers):
        super(CNN_LSTM, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去掉最后的全连接层
        self.lstm = nn.LSTM(cnn_output_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        return r_out[:, -1, :]  # 取最后一个时间步的输出作为特征
    
class DEC(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super(DEC, self).__init__()
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, input_dim))
        torch.nn.init.xavier_normal_(self.cluster_centers)

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2))
        q = q ** ((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    def dec_loss(self, q, p):
        return nn.KLDivLoss(reduction='batchmean')(q.log(), p)


