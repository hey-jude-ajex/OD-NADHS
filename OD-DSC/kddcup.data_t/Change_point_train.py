

import tensorflow as tf


tf.random.set_seed(7)

#if __name__ == "__main__":
import numpy as np
import DSCModel

import pandas as pd
import SpectralCluster
import tensorflow as tf
tf.random.set_seed(7)

class Train(object):
    def __init__(self, data):
        self.data = data

    # print("21", norm_data.shape, type(norm_data))
        self.learning_rate = 0.08
        self.training_epochs_1 = 10
        self.training_epochs_2 = 220

        self.batch_size, self.dim = self.data.shape

        self.network_architecture = dict(n_hidden_enc_1=27,  # 1nd layer encode neurons
                                n_hidden_enc_2=20,  # 2nd layer encode neuron2
                                n_hidden_dec_1=20,  # 1nd layer decode neurons
                                n_hidden_dec_2=27,  # 2nd layer decode neurons
                                n_input=self.dim)  # dimension of data input
        self.alpha_dsc = {
            'a1': 1,
            'a2': 2,
        }

        self.ro = 0.74
        self.d = 27
        self.alpha = 0.96

        self.K = 6
        self.model_path = './model/'

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


import time
import numpy
if __name__ == "__main__":
    change_point_index = str(6)
    data_dir = '../data/'
    time_start = time.time()
    data_dir_e = '../../data/KDDCUP_t/data_progressing/data_' + change_point_index + '_normal_minmax.csv'
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
    data_test = np.array(data_test)
    # 训练数据维数
    dim = data_test.shape[1]
    norm_data = data_test[:, 0:dim - 1]  # normdata是去掉标签的数据 [500, 27]

    Train = Train(norm_data)
    Train.get_Cpoint_train_coef(norm_data, change_point_index)

































