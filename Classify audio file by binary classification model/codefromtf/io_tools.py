"""Input and output helpers to load in data.
"""
import numpy as np


def read_dataset_tf(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature of each sample

        T(numpy.ndarray): class label vector T = [[y1],
                                                  [y2],
                                                  [y3],
                                                   ...]
                             where yi is 1/0, the label of each sample
    """
    ###############################################################
    f = open(path_to_dataset_folder+'/'+index_filename,'r')
    lines = f.readlines()
    index_list = []
    sample_filename_list = []
    feature_list = []
    for i in lines:
        if (i.split(" ")[0]) == "-1":
            index_list.append("0")
        else:
            index_list.append(i.split(" ")[0])
        sample_filename_list.append(i.split(" ")[1].strip())

    T = np.asarray(index_list).reshape(1290,1)
    for i in sample_filename_list:
        filepath = path_to_dataset_folder +'/'+i
        s = open(filepath,'r')
        line = s.readlines()
        for n in line:
            temp_list = []
            temp_list = list(n.strip().split("  "))
            temp_list = [1] + temp_list
            feature_list.append(temp_list)
    A = np.asarray(feature_list)

    return (A.astype(float),T.astype(float))
    ###############################################################


