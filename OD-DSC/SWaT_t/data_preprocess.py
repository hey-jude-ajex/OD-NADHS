import ast
import csv
import os
import sys

from pickle import dump
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler





def load_data(dataset):
    output_folder = '../../data/SWaT/processed'
    os.makedirs(output_folder, exist_ok=True)
    train_df = pd.read_csv("../../data/SWaT/SWaT_Dataset_Normal_v1.csv", delimiter=",")
    train_df_ = train_df.drop(["Timestamp", "Normal/Attack"], axis=1)
    for i in list(train_df_):
        train_df_[i] = train_df_[i].apply(lambda x: str(x).replace(",", "."))
    train_df_ = train_df_.astype(float)
    X_train = train_df_.values
    with open(os.path.join(output_folder, dataset + "_" + "train" + ".pkl"), "wb") as file:
        dump(X_train, file)

    test_df = pd.read_csv("../../data/SWaT/SWaT_Dataset_Attack_v0.csv", delimiter=";")
    #numpy.savetxt("../../data/SWaT/SWaT_Dataset_Attack.csv", test_df, delimiter=',', header=None)


    test_df_ = test_df.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)

    for i in list(test_df_):
        test_df_[i] = test_df_[i].apply(lambda x: str(x).replace(",", "."))
    test_df_ = test_df_.astype(float)
    X_test = test_df_.values
    print("test_df_", test_df_.shape)

    numpy.savetxt("../../data/SWaT/SWaT_Dataset_Attack.csv", X_test, delimiter=',')
    with open(os.path.join(output_folder, dataset + "_" + "test" + ".pkl"), "wb") as file:
        dump(X_test, file)

    y_test = []
    print("45", test_df.shape, test_df['Normal/Attack'].index.shape)
    for index in test_df['Normal/Attack'].index:
        label = test_df['Normal/Attack'].get(index)
        if label == "Normal":
            y_test.append(0)
        elif label == "Attack":

            y_test.append(1)
        else:
            y_test.append(0)
    y_test = np.asarray(y_test)
    print("53", y_test.shape)
    with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
        dump(y_test, file)


def load_data_t(dataset):
    output_folder = '../../data/SWaT/processed'
    os.makedirs(output_folder, exist_ok=True)
    train_df = pd.read_csv("../../data/SWaT/t_train_1.csv", delimiter=",")
    train_df_ = train_df.drop(["Timestamp", "Normal/Attack"], axis=1)
    for i in list(train_df_):
        train_df_[i] = train_df_[i].apply(lambda x: str(x).replace(",", "."))
    #train_df_ = train_df_.astype(float)
    X_train = train_df_.values
    with open(os.path.join(output_folder, dataset + "_" + "train_t_1" + ".pkl"), "wb") as file:
        dump(X_train, file)

    test_df = pd.read_csv("../../data/SWaT/t_1.csv", delimiter=";")

    test_df_ = test_df.drop(["Timestamp", "Normal/Attack"], axis=1)

    for i in list(test_df_):
        test_df_[i] = test_df_[i].apply(lambda x: str(x).replace(",", "."))
    test_df_ = test_df_.astype(float)
    X_test = test_df_.values
    print("test_df_", test_df_.shape)

    numpy.savetxt("../../data/SWaT/SWaT_Dataset_Attack.csv", X_test, delimiter=',')
    with open(os.path.join(output_folder, dataset + "_" + "test_1" + ".pkl"), "wb") as file:
        dump(X_test, file)

    y_test = []
    for index in test_df['Normal/Attack'].index:
        label = test_df['Normal/Attack'].get(index)
        if label == "Normal":
            y_test.append(0)
        elif label == "Attack":

            y_test.append(1)
    y_test = np.asarray(y_test)

    with open(os.path.join(output_folder, dataset + "_" + 'test_label_1' + ".pkl"), "wb") as file:
        dump(y_test, file)


def save_z(z, filename='z'):
    """
    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + '_' + str(i) + '.txt', 'w') as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write('%f ' % (z[j][i][k]))
                file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def preprocess(df):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif dataset == "SWaT":
        return 51
    else:
        raise ValueError('unknown dataset '+str(dataset))


def preprocess(df):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    prefix = '../../data/SWaT/processed'
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("maxsize", max_test_size, max_train_size)
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)
    print("206", test_start, test_end)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    print(test_label)
    n = 0
    for i in range(test_label.shape[0]):
        if test_label[i] == 1:
            n = n+1
    print(n)
    return (train_data, None), (test_data, test_label)

import pandas as pd
import random
def dbscan(data_set, eps, min_pts):
    #eps:邻域半径
    #min_pts:密度阈值
    examples_nus = np.shape(data_set)[0]  # 样本数量
    unvisited = [i for i in range(examples_nus)]  # 未被访问的点
    visited = []  # 已被访问的点
    # cluster为输出结果，表示对应元素所属类别
    # 默认是一个长度为examples_nus的值全为-1的列表，-1表示噪声点
    cluster = [-1 for i in range(examples_nus)]

    k = - 1  # 用k标记簇号，如果是-1表示是噪声点
    while len(unvisited) > 0:  # 只要还有没有被访问的点就继续循环
        p = random.choice(unvisited)  # 随机选择一个未被访问对象
        unvisited.remove(p)
        visited.append(p)

        nighbor = []  # nighbor为p的eps邻域对象集合，密度直接可达
        for i in range(examples_nus):
            if i != p and np.sqrt(np.sum(np.power(data_set[i, :]-data_set[p, :], 2))) <= eps:  # 计算距离，看是否在邻域内
                nighbor.append(i)

        if len(nighbor) >= min_pts:  # 如果邻域内对象个数大于min_pts说明是一个核心对象
            k = k+1
            cluster[p] = k  # 表示p它属于k这个簇

            for pi in nighbor:  # 现在要找该邻域内密度可达
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)

                    # nighbor_pi是pi的eps邻域对象集合
                    nighbor_pi = []
                    for j in range(examples_nus):
                        if np.sqrt(np.sum(np.power(data_set[j]-data_set[pi], 2))) <= eps and j != pi:
                            nighbor_pi.append(j)

                    if len(nighbor_pi) >= min_pts:  # pi是否是核心对象，通过他的密度直接可达产生p的密度可达
                        for t in nighbor_pi:
                            if t not in nighbor:
                                nighbor.append(t)
                if cluster[pi] == -1:  # pi不属于任何一个簇，说明第pi个值未改动
                    cluster[pi] = k
        else:
            cluster[p] = -1  # 不然就是一个噪声点
    return cluster
#def get_result(data, label):

import numpy

if __name__ == "__main__":
    #load_data("SWaT")

    window_length = 12
    train, test = get_data("SWaT")


    X_train_ = train[0]
    X_test_ = test[0]
    y_test_ = test[1]
    X_train_ = preprocess(X_train_)
    X_test_ = preprocess(X_test_)
    print(X_train_.shape)
    X_train = X_train_[np.arange(window_length)[None, :] + np.arange(X_train_.shape[0] - window_length)[:, None]]
    X_test = X_test_[np.arange(window_length)[None, :] + np.arange(X_test_.shape[0] - window_length)[:, None]]
    y_test = y_test_[np.arange(window_length)[None, :] + np.arange(y_test_.shape[0] - window_length)[:, None]]

    X = X_test_[194872:196872]
    #Y = y_test_[281131:282131]
    #print("X.shape", X.shape, Y.shape)
    #X_t = X_train_[281231:282231]
    #X_train_t= X_test_[172952:173511]
    numpy.savetxt("../../data/SWaT/SWaT_normal_10_t.csv", X, delimiter=',')
    #numpy.savetxt("../../data/SWaT/SWaT_C_P_6.csv", X_t, delimiter=',')
    #numpy.savetxt("../../data/SWaT/SWaT_label_6_.csv", Y, delimiter=',')
    #numpy.savetxt("../../data/SWaT/SWaT_C_P_3_t.csv", X_train_t, delimiter=',')
    #label = dbscan(X, 0.4, 27)
    #print(label)
    #print(label)

