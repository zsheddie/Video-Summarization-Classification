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


saved = np.load('E:/KIT/FZI_hiwi/Project/saved_summaries/CLK_decoder_002.npz')
pred = saved['arr1'] 
#p = saved['arr2']
#label = saved['arr3']
#print(p)
#print(label)

class_pred = np.argmax(q, axis=1)
print(class_pred)



# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.plot(class_pred)
plt.plot((20 * label))
#plt.scatter(video_summaries[:, 0], video_summaries[:, 1], c=labels, cmap='viridis')
plt.title('OPTICS Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 输出聚类结果
unique_labels = np.unique(class_pred)
print(f"Clusters found: {len(unique_labels)}")
