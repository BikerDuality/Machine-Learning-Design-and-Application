import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs

# 创建随机分布的，可以分离的400个样本点
X, y = make_blobs(n_samples=400, centers=2, random_state=32)
# 创建LinearSVC对象
clf = LinearSVC(C=1000)
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 画出决策函数
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 网格化评价模型
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
# 画分类边界
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.title("Maximum margin using LinearSVC")
plt.show()
