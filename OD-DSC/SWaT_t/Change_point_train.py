
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


tf.random.set_seed(7)




class Train(object):
    def __init__(self, data):
        self.data = data

        self.learning_rate = 0.02
        self.training_epochs_1 = 20
        self.training_epochs_2 = 200
        self.batch_size, self.dim = self.data.shape


        self.network_architecture = dict(n_hidden_enc_1=45,  # 1nd layer encode neurons
                                n_hidden_enc_2=33,  # 2nd layer encode neuron2
                                n_hidden_dec_1=33,  # 1nd layer decode neurons
                                n_hidden_dec_2=45,  # 2nd layer decode neurons
                                n_input=self.dim)  # dimension of data input
        self.alpha_dsc = {
            'a1': 1,
            'a2': 10,
             }

        self.ro = 0.85
        self.d = 50
        self.alpha = 0.785

        self.K = 6
        self.model_path = './model/'

        #Change_point点到来时进行训练模型的操作
    def get_RMSE(self, y_int, y_out):
        R = np.sqrt(((y_int - y_out) ** 2).mean())
        return R

    def get_Cpoint_train_coef(self, norm_data, C_point_index):
            # change point点处的模型训练
        dsc = DSCModel.DSC(network_architecture=self.network_architecture, learning_rate=self.learning_rate,
                           batch_size=self.batch_size,
                           alpha=self.alpha_dsc, C_point_index=C_point_index,
                           model_path=self.model_path)
        # train model
        DSCModel.train_model(dsc, norm_data, training_epochs=self.training_epochs_2)

import numpy

import time
if __name__ == "__main__":
    change_point_index = str(36)

    data_dir = '../data/'

    # change_point 点训练数据的位置
    time_start = time.time()
    data_dir_e = '../../data/SWaT_t/data_progressing/SWaT_' + change_point_index + '_minmax.csv'

    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
    data_test = np.array(data_test)

    # 训练数据维数
    dim = data_test.shape[1]
    norm_data = data_test[:, 0:dim - 1]  # normdata是去掉标签的数据 [1000, 51]

    Train = Train(norm_data)
    Train.get_Cpoint_train_coef(norm_data, change_point_index)









