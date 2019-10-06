from numpy import *
# 训练朴素贝叶斯模型（仅针对属性值为离散型情况）
class BayesClassifier():   #简单贝叶斯分类器
    def __init__(self):
        pass
    
    # 分离数据集的属性和标签，并分别保存下来
    def getFeatures(self, dataElem, Label):
        self.Label = Label  # 数据集的标签名称
        self.FLists = [ cl for cl in dataElem] # 数据集的属性名称
        self.FLists.remove(self.Label)
        return self.FLists
    
    # 分离数据：数据集和标签
    def splitData(self, dataSets):
        labels = [ cl[self.Label] for cl in dataSets ] # 标签数据集
        features = [] # 属性数据集
        for i in range(len(dataSets)):
            feature = {}
            for fa in dataSets[i]: # 处理每一个数据
                if fa != self.Label: # 判断是标签，还是属性
                    feature[fa] = dataSets[i][fa]
            # print(feature)
            features.append(feature)
        return features, labels
    
    # 训练简单贝叶斯分类器    
    def train(self,features,labels):
        self.sampleNum = len(features)   # 样本数目
        self.countDic = {}               # 统计各个条件概率的出现次数
        self.labelSet = set([])          # 集合存放类标，如：Y=1 or Y=0
        for i in range(len(labels)):  # 统计类标不同值出现的次数
            TempStr = 'Y=' + str(labels[i])
            self.labelSet.add(str(labels[i]))
            if TempStr in self.countDic:
                self.countDic[TempStr] += 1
            else:
                self.countDic[TempStr] = 1 
            for i in range(len(features)): #统计各个条件概率组合出现的次数
               for fl in self.FLists:
                TempStr = 'F' + str(fl) + '=' + str(features[i][fl]) + '|Y=' + str(labels[i])
                if TempStr in self.countDic:
                    self.countDic[TempStr] += 1
                else:
                    self.countDic[TempStr] = 1
        for key in self.countDic.keys():        #遍历次数统计字典计算概率
            if key.find('|') != -1:             #计算条件概率P(Fi=a|Y=b)
                targetStr = key[key.find('|') + 1:]    #类标字符串:  Y=1 or Y=-0
                self.countDic[key] /= self.countDic[targetStr]    #计算条件概率P(Fi=a|Y=b)=Count(Fi=a,Y=b)/Count(Y=b)

        for label in self.labelSet:     #计算类标概率P(Y=b)      
            TempStr = "Y=" + str(label)
            self.countDic[TempStr] /= self.sampleNum

    def classify(self, feature):  #使用训练后的贝叶斯分类器分类新样本
    #   计算后验概率P(Y=b|Sample=feature)
        probabilityMap = {}
        for label in self.labelSet:
            TempProbability = 1.0
            for fl in self.FLists:
                TempStr = 'F' + str(fl) + '=' + str(feature[fl]) + '|Y=' + label
                if TempStr not in self.countDic:   #遇到新的特征值，导致该概率P(Fi=a|Y=b)为0，将它校正为非0值（1/Count(Y=b))
                       TempProbability *= (1.0 / self.countDic['Y=' + label]) / self.sampleNum
                else:
                    TempProbability *= self.countDic[TempStr]
            TempProbability *= self.countDic['Y=' + label]
            probabilityMap[label] = TempProbability
        maxProbability = 0.0
        for label in self.labelSet:     #选取使后验概率P(Y=b|Sample=feature)最大的类标作为目标类标
            if  probabilityMap[label] > maxProbability:
                maxProbability = probabilityMap[label]
                targetLabel = label
        probabilityMap.clear()
        return targetLabel

    def __del__(self):
        self.countDic.clear()
        
if __name__ == "__main__":
    #data
    data = [
{"Outlook":"Sunny","Temp":"Hot","Humidity":"High","Windy":"Weak","class":"No" },
{"Outlook":"Sunny","Temp":"Hot","Humidity":"High","Windy":"Strong","class":"No"},
{"Outlook":"Overcast","Temp":"Hot","Humidity":"High","Windy":"Weak","class":"Yes" },
{"Outlook":"Rain","Temp":"Mild","Humidity":"High","Windy":"Weak","class":"Yes"},
{"Outlook":"Rain","Temp":"Cool","Humidity":"Normal","Windy":"Weak","class":"Yes"},
{"Outlook":"Rain","Temp":"Cool","Humidity":"Normal","Windy":"Strong","class":"No"},
{"Outlook":"Overcast","Temp":"Cool","Humidity":"Normal","Windy":"Strong","class":"Yes"},
{"Outlook":"Sunny","Temp":"Mild","Humidity":"High","Windy":"Weak","class":"No"},
{"Outlook":"Sunny","Temp":"Cool","Humidity":"Normal","Windy":"Weak","class":"Yes"},
{"Outlook":"Rain","Temp":"Mild","Humidity":"Normal","Windy":"Weak","class":"Yes"},
{"Outlook":"Sunny","Temp":"Mild","Humidity":"Normal","Windy":"Strong","class":"Yes"}, 
{"Outlook":"Overcast","Temp":"Mild","Humidity":"High","Windy":"Strong","class":"Yes"},
{"Outlook":"Overcast","Temp":"Hot","Humidity":"Normal","Windy":"Weak","class":"Yes"},
{"Outlook":"Rain","Temp":"Mild","Humidity":"High","Windy":"Strong","class":"No" }]
    #calculate
    NBC = BayesClassifier()
    NBC.getFeatures(data[0], "class")
    #ToDo 获取数据 
    features, labels = NBC.splitData(data)
    NBC.train(features, labels)
    #ToDo 进行训练 
    print(NBC.classify({"Outlook":"Sunny","Temp":"Cool","Humidity":"High", "Windy":"Strong"}))
    print(NBC.classify({"Outlook":"Overcast","Temp":"Cool","Humidity":"Normal", "Windy":"Strong"}))
