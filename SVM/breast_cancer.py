import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_data_classfication():
	'''
	加载用于分类问题的数据集
	:return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
	'''
	iris=datasets.load_breast_cancer()# 使用 scikit-learn 自带的 iris 数据集
	return train_test_split(iris.data, iris.target, test_size=0.25,
		random_state=0,stratify=iris.target) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

def test_SVC(*data):
	'''
	测试 SVC 的用法。这里使用的是最简单的线性核
	:param data:  可变参数。它是一个元组，这里要求其元素依次为训练样本集、测试样本集、训练样本的标记、测试样本的标记
	:return: None
	'''
	X_train,X_test,y_train,y_test=data
	cls=SVC(kernel='linear')
	cls.fit(X_train,y_train)
	# print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
	print('Score: %.2f' % cls.score(X_test, y_test))

if __name__=="__main__":
	X_train,X_test,y_train,y_test=load_data_classfication()
	test_SVC(X_train,X_test,y_train,y_test)
