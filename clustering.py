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
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

k = 6
saved = np.load('E:/KIT/FZI_hiwi/Project/saved_summaries/CLK_decoder_002.npz')
video_summaries = saved['arr1']
print(video_summaries.shape)
## min_samples: 核心对象的最小样本数，较小的min_samples可以找到更小的聚类
## xi: 相邻样本之间的最小距离，xi值越小，不同聚类之间的距离越小，产生的聚类数目会增加
## min_cluster_size：规定了一个聚类最少包含的样本数目
optics = OPTICS(min_samples=4, xi=0.01, min_cluster_size=10) 
optics.fit(video_summaries)

# 获取聚类标签
labels = optics.labels_
print(labels)

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


#############   K-means     ###########
scaler = StandardScaler()
video_summaries = scaler.fit_transform(video_summaries)

kmeans = KMeans(n_clusters=k, random_state=0)
y_kmeans = kmeans.fit_predict(video_summaries)

cluster_labels = y_kmeans.tolist()

# 输出每个样本的簇标签列表
print(cluster_labels)

# 获取质心
centers = kmeans.cluster_centers_

plt.scatter(video_summaries[:, 0], video_summaries[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()