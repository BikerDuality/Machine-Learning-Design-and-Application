import numpy as np


def distance(input, dataset):
    dataset_size = dataset.shape[0]
    diffmat = np.tile(input, (dataset_size, 1))-dataset
    sq_diffmat=diffmat**2
    sq_distances=sq_diffmat.sum(axis=1)
    distances=sq_distances**0.5
    return distances

def classify(input, dataset, labels, k):
    distances = distance(input, dataset)
    sorted_distances_index = distances.argsort()
    class_count = {}
    for index in range(k):
        label = labels[sorted_distances_index[index]]
        class_count[label] = class_count.get(label, 0)+1
    return max(class_count, key=class_count.get)

group=np.array([(1,1),(1,2),(0,1),(0,2)])
labels=['A','A','B','B']
print(classify([0,0],group,labels,3))