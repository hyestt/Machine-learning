from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
my_dir = './data/'
data_file = my_dir + 'iris.data'
data_list = []
with open(data_file, 'r') as f:
    line = f.readline()
    for line in f:
        a = line.strip().split(",")
        data_list.append(a[0:4])

#change the list to np_array and reshape

data_np = np.asarray(data_list).reshape(len(data_list),4)
data_np = data_np.astype(np.float)

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

num_sample = data_np.shape[0]

def k_means(C):
    C = np.array(C)
    #first column store which centroid the sample belong to
    #second column store the error between the sample and centroid
    cluster_ass = np.zeros(num_sample)
    ClusterChange = True
    while ClusterChange:
        ClusterChange = False
        for i in range(num_sample):
            original_distance = 0
            minDistance = 1000
            min_index = 0
            for j in range(k):
                ##calculate the distance**2 between point and centroid
                distance = sum((C[j,:]- data_np[i,:])**2)
                if distance < minDistance:
                    minDistance = distance
                    min_index = j
            #update cluster
            if cluster_ass[i] != min_index:
                ClusterChange = True
                cluster_ass[i] = min_index

        #update centroid
        for j in range(k):
            pointsIncluster = data_np[np.where(cluster_ass == j)]
            C[j,:] = np.mean(pointsIncluster,axis=0)
        C_final = C
    return C_final


print(k_means(C))





