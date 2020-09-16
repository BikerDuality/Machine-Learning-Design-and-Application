from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据集，随后将其分为训练数据集和测试数据集
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, random_state=0, stratify=digits.target)
# 生成学习率数组
learnings = np.linspace(0.01, 1.0, num=25)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
testing_scores = []
training_scores = []
# 针对每一种学习率，训练一个梯度提升分类器，分别在训练数据集和测试数据集上进行评价
for learning in learnings:
    clf = GradientBoostingClassifier(learning_rate=learning)
    clf.fit(X_train, y_train)
    training_scores.append(clf.score(X_train, y_train))
    testing_scores.append(clf.score(X_test, y_test))

ax.plot(learnings, training_scores, label="Training Score")
ax.plot(learnings, testing_scores, label="Testing Score")
ax.set_xlabel("learning_rate")
ax.set_ylabel("score")
ax.legend(loc="lower right")
ax.set_ylim(0, 1.05)
plt.suptitle("GradientBoostingClassifier")
plt.show()
