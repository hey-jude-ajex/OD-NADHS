# from sklearn.manifold import TSNE
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# # 使用t-SNE将数据降维到2D
# path_str = str(1)
# data_path = '../../data/'
# data_dir = '../../data/KDDCUP_t/data_progressing/data_'+ path_str+ '_normal_minmax.csv'
# data = pd.read_csv(data_dir, header=None, dtype=float)
# # 将数据分为特征和标签
# X = data.iloc[:, :-1].values  # 前27列为特征
# Y = data.iloc[:, -1].values   # 第28列为标签
# print(X.shape)
# print(Y)
#
# # 使用t-SNE将数据降维到2D
# tsne = TSNE(n_components=2, random_state=42)
# X_t = tsne.fit_transform(X)
#
# # 可视化
# plt.figure(figsize=(10, 6))
# plt.scatter(X_t[Y == 0, 0], X_t[Y == 0, 1], label='Normal', alpha=0.5)
# plt.scatter(X_t[Y == -1, 0], X_t[Y == -1, 1], label='Outliers -1', alpha=0.5, color='red')
# plt.scatter(X_t[Y == -2, 0], X_t[Y == -2, 1], label='Outliers -2', alpha=0.5, color='blue')
# plt.scatter(X_t[Y == -3, 0], X_t[Y == -3, 1], label='Outliers -3', alpha=0.5, color='green')
# plt.legend()
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.title('t-SNE Visualization of Outliers')
# plt.show()



from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
path_str = str(1)
data_path = '../../data/'
data_dir = '../../data/KDDCUP_t/data_progressing/data_'+ path_str+ '_normal_minmax.csv'
data = pd.read_csv(data_dir, header=None, dtype=float)

# 将数据分为特征和标签
X = data.iloc[:, :-1].values  # 前27列为特征
Y = data.iloc[:, -1].values   # 第28列为标签


tsne = TSNE(n_components=2, random_state=42)
X_t = tsne.fit_transform(X)
print(np.array(X_t).shape)

# 选择两个特征进行可视化
feature1 = 0  # 第一个特征列的索引
feature2 = 1  # 第二个特征列的索引

plt.figure(figsize=(10, 6))
plt.scatter(X_t[Y == 0, 0], X_t[Y == 0, 1], label='Normal', alpha=0.5)
plt.scatter(X_t[Y == -1, 0], X_t[Y == -1, 1], label='Outliers -1', alpha=0.5, color='red')
plt.scatter(X_t[Y == -2, 0], X_t[Y == -2, 1], label='Outliers -2', alpha=0.5, color='blue')
plt.scatter(X_t[Y == -3, 0], X_t[Y == -3, 1], label='Outliers -3', alpha=0.5, color='green')

# plt.scatter(X[Y == 0, feature1], X[Y == 0, feature2], label='Normal', alpha=0.5)
# plt.scatter(X[Y == -1, feature1], X[Y == -1, feature2], label='Outliers -1', alpha=0.5, color='red')
# plt.scatter(X[Y == -2, feature1], X[Y == -2, feature2], label='Outliers -2', alpha=0.5, color='blue')
# plt.scatter(X[Y == -3, feature1], X[Y == -3, feature2], label='Outliers -3', alpha=0.5, color='green')
plt.legend()
plt.xlabel(f'Feature {feature1 + 1}')
plt.ylabel(f'Feature {feature2 + 1}')
plt.title('Scatter Plot of Original Features')
plt.show()
