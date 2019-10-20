import os
import jieba
import random
from sklearn.naive_bayes import MultinomialNB


def TextImport(folder_path):
    data_list=[]
    class_list=[]
    folder_list = os.listdir(folder_path)
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)  # 根据子文件夹，生成新的路径
        files = os.listdir(new_folder_path)  # 存放子文件夹下的txt文件的列表
        for file in files:
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:  # 打开txt文件
                raw = f.read()
            word_cut = jieba.cut(raw, cut_all=False)  # 精简模式，返回一个可迭代的generator
            word_list = list(word_cut)  # generator转换为list
            data_list.append(word_list)							#添加数据集数据
            class_list.append(folder)							#添加数据集类别
    return data_list,class_list

def TextFeatures(data_list, feature_words):
	def text_features(text, feature_words):						#出现在特征集中，则置1
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words]
		return features
	feature_list = [text_features(text, feature_words) for text in data_list]
	return feature_list				#返回结果

def feature_words_dict(all_words_list, deleteN, stopwords_set = set()):
	feature_words = []							#特征列表
	n = 1
	for t in range(deleteN, len(all_words_list), 1):
		if n > 1000:							#feature_words的维度为1000
			break
		#如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
		if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
			feature_words.append(all_words_list[t])
		n += 1
	return feature_words