
import time

import numpy as np
import DSCModel
from sklearn import cluster
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
import Partial_training_online
from sklearn.metrics import mean_squared_error
import SpectralCluster
# from sklearn.clustering import SpectralClustering
from sklearn.cluster import SpectralClustering
import tensorflow as tf
import scipy.io as sio
import Best_map

tf.random.set_seed(7)

#if __name__ == "__main__":
import numpy as np
import DSCModel
import DSC_t

import DSCModel_NChange
import Exception_handling
import pandas as pd
import SpectralCluster
import tensorflow as tf
tf.random.set_seed(7)


data_dir = '../../data/'
# data_dir_e = '../data/test_data/test_normal_result_minmax_1.csv'
data_dir_e = '../../data/KDDCUP_t/data_progressing/data_1_normal_minmax.csv'
print("data_dir: ", data_dir)
data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
data_test = np.array(data_test)
data_dim = data_test.shape[1]-1
print(data_dim)
data = data_test[:, 0:data_dim]

#print("21", norm_data.shape, type(norm_data))
learning_rate = 0.08
training_epochs_1 = 10
training_epochs_2 = 220

batch_size, dim = data.shape

network_architecture = dict(n_hidden_enc_1=27,  # 1nd layer encode neurons
                                n_hidden_enc_2=20,  # 2nd layer encode neuron2
                                n_hidden_dec_1=20,  # 1nd layer decode neurons
                                n_hidden_dec_2=27,  # 2nd layer decode neurons
                                n_input=dim)  # dimension of data input
alpha_dsc = {
        'a1': 1,
        'a2': 2,
    }
#ro = 0.7
#ro = max(0.4 - (5 - 1) / 10 * 0.1, 0.1)
ro = 0.74
d = 27
alpha =0.96


K = 6
model_path = './model/'

def get_RMSE(y_int, y_out):
    #R = np.sqrt(mean_squared_error((y_int, y_out)))
    R = np.sqrt(((y_int-y_out)**2).mean())
    return R

def get_Cpoint_train_coef(norm_data, C_point_index):
    #change point点处的模型训练
    dsc = DSCModel.DSC(network_architecture=network_architecture, learning_rate=learning_rate, batch_size=batch_size,
                       alpha=alpha_dsc, C_point_index=C_point_index,
                       model_path=model_path)
    # train model
    DSCModel.train_model(dsc, norm_data, training_epochs=training_epochs_2)
    #y_intput, coef = DSCModel.coef_get(model_path, norm_data, C_point_index)
    # print("coef", coef)

    #C = SpectralCluster.thrC(coef, ro)
    #grp, L= SpectralCluster.post_proC(C, K, d, alpha)
    # print("62", L.shape, type(L))
    #return y_intput, L, grp

import time
import numpy
if __name__ == "__main__":
    change_point_index = str(29)
    # time.sleep(10)
    print("------------------------------------------------第", change_point_index,
          "批数据------------------------------------------------------")
    print("time：", time.time())
    data_dir = '../data/'
    # data_dir_e = '../data/test_data/test_normal_result_minmax_1.csv'

    #print("42", change_point_index)
    time_start = time.time()
    # change_point 点训练数据的位置
    #for i in range (3):
    data_dir_e = '../../data/KDDCUP_t/data_progressing/data_' + change_point_index + '_normal_minmax.csv'
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
    data_test = np.array(data_test)
    # 训练数据维数
    dim = data_test.shape[1]
    #print("source_data_shape", data_test.shape)
    norm_data = data_test[:, 0:dim - 1]  # normdata是去掉标签的数据 [500, 27]

    get_Cpoint_train_coef(norm_data, change_point_index)
    time_end = time.time()
    time_sum = time_end - time_start
    print("C_P_time_sum", time_sum)
    #print("---------132-------------", Y.shape, norm_data.shape, type(Y), type(norm_data))
    #R = get_RMSE(norm_data, np.array(Y))

    #print("---------------------------grp---------------------------")
    #print(grp, len(grp))
    #print("数据训练后的RMSE:", R)



































