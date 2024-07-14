import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs

# 生成示例数据
centers = [[1, 1], [5, 5], [3, 10]]
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=42)

# 执行OPTICS聚类
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics.fit(X)

# 获取聚类标签
labels = optics.labels_

# 获取核心距离和可达距离
core_distances = optics.core_distances_
reachability_distances = optics.reachability_

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('OPTICS Clustering')
plt.show()
