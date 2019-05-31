import sys
import os
from six.moves import urllib
from scipy.io import loadmat
import logging
from utils.adapt_data import adapt_labels_outlier_task
import numpy as np
from sklearn.model_selection import train_test_split
from chest_src import utils

logger = logging.getLogger(__name__)

#instead of label, there will be dataset 0  or dataset 1. dataset 0 will involve  the first  set of holdouts, dataset  1 the second.
#  we will use hardcoded dictionaries to achieve this.

def get_train(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("train", label, centered, normalize)

def get_test(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("test", label, centered, normalize)
    
def get_valid(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("valid", label, centered, normalize)

def get_shape_input():
    return (None, 224, 224, 3)

def get_shape_input_flatten():
    return (None, 150528)

# abnormal_list = ["Lung", "Spine", "Pulmonary Atelectasis", "Cardiomegaly"] 0
# abnormal_list = ["Calcinosis", "Opacity", "Thoracic Vertebrae", "Calcified Granuloma"] 1

# figure out how this is used with the split and all!
def _get_adapted_dataset(split, label, centered, normalize):
    """
    Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
            mode (str): inlier or outlier
            label (int): int in range 0 to 10, is the class/digit
                         which is considered inlier or outlier
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns :
            (tuple): <training, testing> images and labels
            :type label: object
    """
    dataset = {}

    if label == 0:
        abnormal_list = ["Lung", "Spine", "Pulmonary Atelectasis", "Cardiomegaly"]
    elif label == 1:
        abnormal_list = ["Calcinosis", "Opacity", "Thoracic Vertebrae", "Calcified Granuloma"]
    
    dataset['x_train'], dataset['x_valid'], dataset['x_test'], dataset['y_train'], dataset['y_valid'], dataset['y_test'] = _load_holdout_dataset(abnormal_list)

    if normalize:
        dataset['x_train'] = dataset['x_train'].astype(np.float32) / 255.0
        dataset['x_valid'] = dataset['x_valid'].astype(np.float32) / 255.0
        dataset['x_test'] = dataset['x_test'].astype(np.float32) / 255.0
    if centered:
        dataset['x_train'] = dataset['x_train'].astype(np.float32) * 2. - 1.
        dataset['x_valid'] = dataset['x_valid'].astype(np.float32) * 2. - 1.
        dataset['x_test'] = dataset['x_test'].astype(np.float32) * 2. - 1.

    key_img = 'x_' + split
    key_lbl = 'y_' + split

    return (dataset[key_img], dataset[key_lbl])

def _load_holdout_dataset(abnormal_list=[]):
    
    """# load data
    X, Y = utils.load_X_and_Y()
    x_train, x_dev, x_test = X
    y_train, y_dev, y_test = Y

    # holdout training and dev data if requested
    X_descr = utils.load_X_descr()
    x_train_all_descr, x_dev_all_descr, x_test_descr = X_descr

    include_x_train = []
    include_y_train = []
    for idx in range(len(x_train)):
        if x_train_all_descr[idx] not in abnormal_list:
            include_x_train.append(x_train[idx])
            include_y_train.append(y_train[idx])

    include_x_dev = []
    include_y_dev = []
    for idx in range(len(x_dev)):
        if x_dev_all_descr[idx] not in abnormal_list:
            include_x_dev.append(x_dev[idx])
            include_y_dev.append(y_dev[idx])
    
    x_train = np.array(include_x_train).reshape((-1, 224, 224, 3))
    y_train = np.array(include_y_train).reshape((-1, 1))
    y_train = np.array(np.zeros(y_train.shape))

    x_dev = np.array(include_x_dev).reshape((-1, 224, 224, 3))
    y_dev = np.array(include_y_dev).reshape((-1, 1))
    y_dev = np.array(np.zeros(y_dev.shape))

    for i in range(len(y_test)):
        if x_test_descr[i] in abnormal_list:
            y_test[i] = 1
        else:
            y_test[i] = 0

    print("average value of  train label: ", np.average(y_train))
    print("average value of  dev label: ", np.average(y_dev))
    print("average value of  test label: ", np.average(y_test))
    return (x_train, x_dev, x_test, y_train, y_dev, y_test)"""

    # load data
    X, Y = utils.load_X_and_Y()
    x_train, x_dev, x_test = X
    y_train, y_dev, y_test = Y

    # holdout training and dev data if requested
    X_descr = utils.load_X_descr()
    x_train_all_descr, x_dev_all_descr, x_test_descr = X_descr

    include_x_train = []
    include_y_train = []
    holdout_x_train = []
    holdout_y_train = []
    for idx in range(len(x_train)):
        if x_train_all_descr[idx] not in abnormal_list:
            include_x_train.append(x_train[idx])
            include_y_train.append(y_train[idx])
        else:
            holdout_x_train.append(x_train[idx])
            holdout_y_train.append(1)

    include_x_dev = []
    include_y_dev = []
    holdout_x_dev = []
    holdout_y_dev = []
    for idx in range(len(x_dev)):
        if x_dev_all_descr[idx] not in abnormal_list:
            include_x_dev.append(x_dev[idx])
            include_y_dev.append(y_dev[idx])
        else:
            holdout_x_dev.append(x_dev[idx])
            holdout_y_dev.append(1)
    
    x_train = np.array(include_x_train).reshape((-1, 224, 224, 3))
    y_train = np.array(include_y_train).reshape((-1, 1))
    y_train = np.array(np.zeros(y_train.shape))

    x_dev = np.array(include_x_dev).reshape((-1, 224, 224, 3))
    y_dev = np.array(include_y_dev).reshape((-1, 1))
    y_dev = np.array(np.zeros(y_dev.shape))

    # add excluded abnormalities (train/dev) to test set!
    for i in range(len(y_test)):
        if x_train_all_descr[i] in abnormal_list:
            y_test[i] = 1
        else:
            y_test[i] = 0

    x_test += holdout_x_train + holdout_x_dev
    y_test += holdout_y_train + holdout_y_dev

    print("average value of  train label: ", np.average(y_train))
    print("average value of  dev label: ", np.average(y_dev))
    print("average value of  test label: ", np.average(y_test))
    return (x_train, x_dev, x_test, y_train, y_dev, y_test)

# abnormal_list = ["Lung", "Spine", "Pulmonary Atelectasis", "Cardiomegaly"] 0
# abnormal_list = ["Calcinosis", "Opacity", "Thoracic Vertebrae", "Calcified Granuloma"] 1
