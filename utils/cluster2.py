from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

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
