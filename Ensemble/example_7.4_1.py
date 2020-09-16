import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

n_estimators = 400
learning_rate = 1
X, y = make_hastie_10_2(n_samples=12000, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 构造基学习器（决策分类树），训练，并计算误差
dt_base = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_base.fit(X_train, y_train)
dt_base_err = 1.0 - dt_base.score(X_test, y_test)

# 构造决策分类树，训练，并计算误差
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

# 构造AdaBoost（SAMME）集成学习器，训练
ada_discrete = AdaBoostClassifier(base_estimator=dt_base, learning_rate=learning_rate, n_estimators=n_estimators, algorithm='SAMME')
ada_discrete.fit(X_train, y_train)

# 构造AdaBoost（SAMME.R）集成学习器，训练
ada_real = AdaBoostClassifier(base_estimator=dt_base, learning_rate=learning_rate, n_estimators=n_estimators, algorithm='SAMME.R')
ada_real.fit(X_train, y_train)

# 计算estimator在数据集上的误差，并输出
def plot_AdaBoost_Error(ax,estimator, X_, y_, labels, color):
    ada_errors = np.zeros((len(estimator),))
    for i, y_pred in enumerate(estimator.staged_predict(X_)):
        ada_errors[i] = zero_one_loss(y_pred, y_)  #计算ero_one_loss
    ax.plot(np.arange(n_estimators) + 1, ada_errors, label=labels, color=color)
# 比较每个学习器的误差
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, n_estimators], [dt_base_err] * 2, 'k-', label='Decision Base Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--', label='Decision Tree Error')
plot_AdaBoost_Error(ax, ada_discrete, X_test, y_test, "Discrete AdaBoost Test Error","red")
plot_AdaBoost_Error(ax, ada_discrete, X_train, y_train, "Discrete AdaBoost Train Error","blue")
plot_AdaBoost_Error(ax, ada_real, X_test, y_test, 'Real AdaBoost Test Error','orange')
plot_AdaBoost_Error(ax, ada_real, X_train, y_train, 'Real AdaBoost Train Error','green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
plt.show()
