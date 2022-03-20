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

    
    sum_dataset = pd.concat([train_x, test_x])
    sum_dataset = sum_dataset.fillna(sum_dataset.mean()).to_numpy(dtype=np.float32)
    sum_target = pd.concat([train_target, test_target]).to_numpy(dtype=np.float32)
    # sum_target = sum_target.fillna(sum_target.mean()).to_numpy(dtype=np.float32)
    
    
    num_classes = len(np.unique(sum_target))
    # sum_dataset = normalize(sum_dataset)

    return sum_dataset, sum_target, num_classes


def split_raw_and_test(data, target):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    data = np.array(data)
    target = np.array(target)

    for train_index, test_index in sss.split(data, target):
        return data[train_index], target[train_index], data[test_index], target[test_index]
        

# only use
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

# v2. use k_fold to get the whole tran val test set

def normalize_test_set(test_data, train_data):
    mean = train_data.mean()
    var = train_data.var()

    return (test_data-mean)/var

def k_fold(data, target):
    skf = StratifiedKFold(5, shuffle=True)
    
    train_sets = []
    train_targets = []

    val_sets = []
    val_targets = []

    test_sets = []
    test_targets = []

    for raw_index, test_index in skf.split(data, target):
        raw_set = data[raw_index]
        raw_target = target[raw_index]

        test_sets.append(data[test_index])
        test_targets.append(target[test_index])

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))

        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

    return np.array(train_sets), np.array(train_targets), np.array(val_sets), np.array(val_targets), np.array(test_sets), np.array(test_targets)





    
        



# TODO 优化五折验证的代码

if __name__ == '__main__':

    pass
