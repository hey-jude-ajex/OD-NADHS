import numpy as np
import DSCModel
import utilss
from sklearn import cluster
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
import SpectralCluster
import SpectralCP
#from sklearn.clustering import SpectralClustering
from sklearn.cluster import SpectralClustering
import tensorflow as tf
import scipy.io as sio
import Best_map

if __name__ == "__main__":
    data_dir = '../data/'
    #data_dir_e = '../data/test_data/test_normal_result_minmax_1.csv'
    data_dir_e = '../data/data_cut/test_1.csv'
    print("data_dir: ", data_dir)
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)

    data_test = np.array(data_test)
    print("source_data_shape", data_test.shape)
    print(data_test[:, 41])
    print(data_test)
    norm_data = data_test[:, 0:41]
    print("21", norm_data.shape, type(norm_data))

    #batch_size = 3200
    learning_rate = 0.1
    training_epochs = 300
    n_hidden = [50]
    batch_size, dim= norm_data.shape
    dim = norm_data.shape[1]
    #ori_data_x = ori_data_x[:batch_size]
    '''
   
    ori_data_x = sio.loadmat('../data/COIL100.mat')
    Img = ori_data_x['fea']
    Label = ori_data_x['gnd']
    norm_data = np.reshape(Img,(Img.shape[0],32,32,1))
    norm_data = norm_data[:batch_size]
     '''
    #print(Img.shape[0])
    #norm_data = ori_data_x.reshape([batch_size, 28, 28, 1])
    #print(ori_data_x[1])
    #print("ori_data_x", norm_data.shape)
    #plt.matshow(norm_data[0])
    #plt.show()
    # Normalization
    #norm_data, norm_parameters = utilss.normalization(ori_data_x)
    #no, dim = ori_data_x.shape

    
    network_architecture = dict(n_hidden_enc_1=30,  # 1nd layer encode neurons
                                n_hidden_enc_2=20,  # 2nd layer encode neurons
                                n_hidden_dec_1=20,  # 1nd layer decode neurons
                                n_hidden_dec_2=30,  # 2nd layer decode neurons
                                n_input=dim)  # dimension of data input


    alpha_dsc = {
        'a1': 1,
        'a2': 15,
    }

    # ############DACIN Model##########################
    #加入DSC层的实验
    # train
    model_path = './model/'
    print("model_payh:", model_path)
    dsc = DSCModel.DSC(network_architecture=network_architecture, learning_rate=learning_rate, batch_size=batch_size, alpha=alpha_dsc,
                       model_path=model_path)
    # train model
    DSCModel.train_model(dsc, norm_data, training_epochs=training_epochs)

    z, z_c, coef = DSCModel.coef_get(model_path, norm_data)
    #print("coef", coef)
    ro = 1
    alpha = 15
    K = 3
    d = network_architecture['n_hidden_enc_2']
    C = SpectralCluster.thrC(coef, ro)
    L = SpectralCluster.post_proC(C, K, d, alpha)
    print("相关性矩阵：", L, L.shape)

    #print("c_sum", c_sum.shape, c_sum)
    #print("argmax", np.argmax(L, axis=0), L.shape)
    # spectral clustering
    #d = ori_data_x.shape[1]
    d = 1.1e-05
    #聚簇测试
    n = 0      #簇标签
    clusters = []
    #c_data = norm_data
    c_data = data_test
    #while norm_data.any():
    n = n+1


        #print("center_index", center_index)
    index_t = [1, 2, 3]
    print("102", type(index_t))
    index_out = []
    for i in range(L.shape[0]):
        index_out.append(i)
    print("106", len(index_out))
    #index_out = np.array(index_out).astype(int)
    print("106", type(index_out))
    k = L.shape[0]
    c = np.zeros(L.shape[0])
    while len(index_out) != 0:
        #print("111", len(index_out))
        c_sum = np.sum(L, axis=1)  # 将每一行的元素相加,将矩阵压缩为一列
        center_index = np.argmax(c_sum)
        #print("center_index", center_index)
        #print("107", L.shape[0])
        index = []
        clusters_t = []
        for i in index_out:
            #print("114", i)
            if L[center_index, i] > d:
                clusters_t.append(c_data[i])
                #print("116", len(index_out))
                #del index_out[i]
                L[i] = c
                #print("123", L[i])
                index_out.remove(i)
                print("125", len(index_out))
                index.append(i)
                #print("104", L.shape)
        #print("index", index)
        #print("cluster_t", len(clusters_t))
        #L = np.delete(L, index, axis=0)
        #print("L.shape", L.shape)
        print("cluusters_t", np.array(clusters_t).shape)
        print("label", np.array(clusters_t)[:, 41])
        if clusters_t  == None:
            outers = []
            for i in index_out:
                del index_out[i]
                outers.append(c_data[i])
            break
        #print("128", len(index_out))
                        #当前未被划分的数据下标
        #print("124", index, type(index))


        #c_data = np.delete(c_data, index, axis=0)
        #print("109", L.shape)
        clusters.append(clusters_t)
    #print("146", len(clusters[1]))
    #clusters.append(c_data)
    print("-----------------------------------------------------------------------------------")
    #print(clusters)































