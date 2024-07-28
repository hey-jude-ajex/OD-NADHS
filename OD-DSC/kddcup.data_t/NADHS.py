import time
import tensorflow as tf
import numpy as np
import pandas as pd
import Partial_training_online
import warnings
import Change_point_parameter
import cluster_detection
import SpectralCluster
import DSCModel_NChange
warnings.filterwarnings("ignore")
tf.random.set_seed(7)


class NADHS(object):
    def __init__(self, threshold_R, norm_data, path_str, C_point_index, parameter, moth_path):
        self.threshold_R = threshold_R
        self.norm_data = norm_data
        self.path_str = path_str
        self.C_point_index = C_point_index
        self.parameter = parameter
        self.moth_path = moth_path

    def get_partal_train(self):
        # 向线上部分输出训练的模型
        # -----------------------------------------------------------------------------------
        dim = self.norm_data.shape[1]  # norm_data:[batchsize,27]
        #parameter
        learning_rate = 0.08
        training_epochs_1 = 50
        network_architecture = dict(n_hidden_enc_1=27,
                                    # 1nd layer encode neurons
                                    n_hidden_enc_2=20,
                                    # 2nd layer encode neuron2
                                    n_hidden_dec_1=20,
                                    # 1nd layer decode neurons
                                    n_hidden_dec_2=27,
                                    # 2nd layer decode neurons
                                    n_input=dim)  # dimension of data input
        alpha_dsc = {
            'a1': 1,
            'a2': 2,
        }
        d = 27
        K = 4
        ro = parameter['ro'][int(self.path_str) - 1]
        alpha = parameter['alpha'][int(self.path_str) - 1]

        # DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate, batch_size=batch_size,
        #                                    alpha=alpha_dsc, batch_num=batch_num, C_point_index = C_point_index,
        #                                    model_path=model_path)
        # print("Off57", type(batch_num), batch_num)
        # DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
        y_intput, coef = DSCModel_NChange.coef_get_NChange(self.moth_path, self.norm_data, self.C_point_index, self.path_str)
        time_s = time.time()
        C = SpectralCluster.thrC(coef, ro)
        # 相关性矩阵
        L = SpectralCluster.post_proC(C, K, d, alpha)
        time_e = time.time()
        print("time_C", time_e - time_s)
        return y_intput, L
    def Change_point_detection(self):

