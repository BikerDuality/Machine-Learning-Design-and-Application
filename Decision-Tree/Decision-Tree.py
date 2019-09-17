import numpy as np
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0)+1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt
def CreateDataSet():
    dataSet = [[1, 1, ' Yes'],
               [1, 1, ' Yes'],
               [1, 0, 'No'],
               [0, 1, 'No'],
               [0, 1, 'No']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
myDat, labels = CreateDataSet()
print(calcShannonEnt(myDat))
