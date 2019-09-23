import numpy as np
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():     
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob * np.math.log(prob,2) #log base 2
    return shannonEnt
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)       #get a set of unique values
        subDataSet=[item[:i+1] for item in dataSet]#sub-dataset for split info calculation
        splitInfo=calcShannonEnt(subDataSet)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = (baseEntropy - newEntropy)/splitInfo     #
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]         #
    if len(dataSet[0]) == 1:    #
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    subLabels=labels[:]
    del(subLabels [bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree         
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
# def createDataSet():
#     dataSet = [[1, 1, ' Yes'],
#                 [1, 1, ' Yes'],
#                 [1, 0, 'No'],
#                 [0, 1, 'No'],
#                 [0, 1, 'No']]
#     labels = ['no surfacing', 'flippers']
#     return dataSet, labels
def createDataSet():
    dataSet=np.loadtxt('lenses.txt',dtype=bytes,delimiter='\t').astype(str).tolist()
    labels = ['1', '2','3','4','5']
    return dataSet, labels
myDat, labels = createDataSet()
print(calcShannonEnt(myDat))
dataSet, labels=createDataSet()
tree=createTree(dataSet, labels)
print(tree)
