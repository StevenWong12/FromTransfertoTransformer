import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
import numpy as np
import os

UCR_PATH = '/dev_data/zzj/hzy/datasets/UCR'

def load_data(dataroot, dataset):
    train = pd.read_csv(os.path.join(dataroot, dataset, dataset+'_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(dataroot, dataset, dataset+'_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    sum_dataset = pd.concat([train_x, test_x]).fillna(0).to_numpy(dtype=np.float32)
    sum_target = pd.concat([train_target, test_target]).fillna(0).to_numpy(dtype=np.float32)

    num_classes = len(np.unique(sum_target))
    # sum_dataset = normalize(sum_dataset)

    return sum_dataset, sum_target, num_classes


def split_raw_and_test(data, target):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    
    for train_index, test_index in sss.split(data, target):
        return np.array(list(map(lambda x :data[x], train_index))), np.array(list(map(lambda x :target[x], train_index))),np.array(list(map(lambda x:data[x], test_index))),np.array(list(map(lambda x:target[x], test_index)))
    
        

def normalize(dataset):
    normalizer = preprocessing.StandardScaler()
    return normalizer.fit_transform(dataset)


def get_k_fold(data, target):
    skf = StratifiedKFold(5, shuffle=True)
    train_sets = []
    val_sets = []
    for train_index, val_index in skf.split(data, target):
        train_sets.append(train_index)
        val_sets.append(val_index)

    val_data = []
    val_target = []
    for val in val_sets:
        val_x = np.array(list(map(lambda x: data[x], val)))
        val_y = np.array(list(map(lambda x: target[x], val)))

        val_data.append(val_x)
        val_target.append(val_y)


    training_data = []
    training_target = []
    for train in train_sets:
        train_x = np.array(list(map(lambda x: data[x], train)))
        train_y = np.array(list(map(lambda x: target[x], train)))

        training_data.append(train_x)
        training_target.append(train_y)

    return training_data, training_target, val_data, val_target
    

if __name__ == '__main__':

    pass
