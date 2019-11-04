from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
Y = iris.target

X,x_test,Y,y_test=train_test_split(X,Y,test_size=0.25,stratify=Y)
# x_test,y_test=X,Y
# 构造三个基学习器和三个集成学习器
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf1 = BaggingClassifier(base_estimator=clf1,n_estimators=30)
eclf2 = BaggingClassifier(base_estimator=clf2, n_estimators=30)
eclf3 = BaggingClassifier(base_estimator=clf3, n_estimators=30)
rfclf = RandomForestClassifier(n_estimators=30)

# 训练学习器
clf1.fit(X, Y)
clf2.fit(X, Y)
clf3.fit(X, Y)
eclf1.fit(X, Y)
eclf2.fit(X, Y)
eclf3.fit(X, Y)
rfclf.fit(X,Y)

# 生成数据网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# 绘图
f, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(10+5, 8))
for idx, clf, tt in zip(list(product([0, 1, 2], [0, 1]))+[(0,2)],
                        [clf1, eclf1, clf2, eclf2, clf3, eclf3,rfclf],
                        ['Decision Tree (depth=4)', 'Bagging DT',
                         'KNN (k=7)','Bagging KNN',
                         'Kernel SVM', 'Bagging SVC',
						 'Radom Forest']):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=Y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt+' '+str(clf.score(x_test,y_test))[:5])

plt.show()
