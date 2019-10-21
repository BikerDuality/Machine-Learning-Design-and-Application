import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import datasets,linear_model,svm
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()

X_train = iris.data #data 表示数据
y_train = iris.target
x=X_train[:, :2]
y=y_train

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0,stratify=y_train)
print(x_train[0:1])
print(y_train[0:1])

cls=svm.LinearSVC()
cls.fit(x_train,y_train)
print('各特征权重：%s,截距:%s'%(cls.coef_,cls.intercept_))
print("算法评分：%.2f" % cls.score(x_test,y_test))

# 开始画图
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
# 生成网格采样点
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
# 测试点
grid_test = np.stack((x1.flat, x2.flat), axis=1)
print('grid_test:\n', grid_test)
# 输出样本到决策面的距离
z = cls.decision_function(grid_test)
print('the distance to decision plane:\n', z)
iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'

# 预测分类值
grid_hat = cls.predict(grid_test)
print('grid_hat:\n', grid_hat)
# reshape grid_hat和x1形状一致
grid_hat = grid_hat.reshape(x1.shape)

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
# 样本点
plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)
# 测试点
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)
plt.xlabel(iris_feature[0], fontsize=20)
plt.ylabel(iris_feature[1], fontsize=20)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('svm in iris data classification', fontsize=30)
plt.grid()
plt.show()
