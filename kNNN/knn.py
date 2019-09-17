import numpy as np

def autoNom(dataset):
    dataset_min=dataset.min(0)
    dataset_max=dataset.max(0)
    dataset_range=dataset_max-dataset_min
    norm_dataset=np.zeros(dataset.shape)
    norm_dataset=dataset-np.tile(dataset_min,(dataset.shape[0],1))
    norm_dataset=norm_dataset/np.tile(dataset_range,(dataset.shape[0],1))
    return norm_dataset

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

dataset_split_rate=0.9
k_neighbor=3
dataset=np.loadtxt('datingTestSet2.txt',delimiter='\t')
dataset=autoNom(dataset)
train_data=dataset[:int(len(dataset)*dataset_split_rate),:-1]
train_label=dataset[:int(len(dataset)*dataset_split_rate),-1]
test_data=dataset[int(len(dataset)*dataset_split_rate):,:-1]
test_label=dataset[int(len(dataset)*dataset_split_rate):,-1]
true_count=0
for index in range(len(test_data)):
    result=classify(test_data[index],train_data,train_label,k_neighbor)
    if result==test_label[index]:
        true_count+=1
accuracy_rate=true_count/len(test_data)
print(accuracy_rate)
# print(classify([0,0],group,labels,3))