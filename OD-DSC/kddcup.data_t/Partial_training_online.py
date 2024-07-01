import numpy as np
import DSCModel


import DSCModel_NChange


import pandas as pd
import SpectralCluster
import tensorflow as tf
import time
tf.random.set_seed(7)


data_dir = '../../data/'
# data_dir_e = '../data/test_data/test_normal_result_minmax_1.csv'
data_dir_e = '../../data/KDDCUP_t/data_progressing/data_1_normal_minmax.csv'
#print("data_dir: ", data_dir)
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


K = 4
model_path = './model/'



#非Change_point点数据的操作
def get_coef(norm_data, batch_num, C_point_index):

    dsc = DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate, batch_size=batch_size,
                                       alpha=alpha_dsc, batch_num=batch_num,
                                       model_path=model_path)
    DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
    y_intput, coef = DSCModel_NChange.coef_get_NChange(dsc, model_path, norm_data)

    C = SpectralCluster.thrC(coef, ro)
    #C = SpectralCluster.S(coef, ro)
    #相关性矩阵
    grp, L= SpectralCluster.post_proC(C, K, d, alpha)

    return y_intput, L

def get_coef_NChange_test(norm_data, batch_num, C_point_index, parameter):
    #向线上部分输出训练的模型
    #-----------------------------------------------------------------------------------
    dim = norm_data.shape[1]        #norm_data:[batchsize,27]
    learning_rate = 0.08
    training_epochs_1 = 50
    network_architecture = dict(n_hidden_enc_1=parameter['n_hidden_enc_1'],  # 1nd layer encode neurons
                                n_hidden_enc_2=parameter['n_hidden_enc_2'],  # 2nd layer encode neuron2
                                n_hidden_dec_1=parameter['n_hidden_dec_2'],  # 1nd layer decode neurons
                                n_hidden_dec_2=parameter['n_hidden_enc_1'],  # 2nd layer decode neurons
                                n_input=dim)  # dimension of data input
    alpha_dsc = {
        'a1':parameter['alpha_dsc_a1'],
        'a2':parameter['alpha_dsc_a2'],
    }
    # ro = 0.7
    # ro = max(0.4 - (5 - 1) / 10 * 0.1, 0.1)
    ro = parameter['ro']
    d = parameter['d']
    alpha = parameter['alpha']



    y_intput, coef = DSCModel_NChange.coef_get_NChange(model_path, norm_data, C_point_index, batch_num)
    #d = network_architecture['n_hidden_enc_2']
    time_s = time.time()

    C = SpectralCluster.S(coef, ro)
    #C = SpectralCluster.thrC(coef, ro)
    # #相关性矩阵
    #
    #L= SpectralCluster.post_proC(C, K, d, alpha)
    time_e = time.time()
    print("time_C", time_e-time_s)
    return y_intput, C

#Change_point点到来时进行训练模型的操作
#C_point_index:Chang point 的数据编号


def get_RMSE(y_int, y_out):
    R = np.sqrt(((y_int - y_out) ** 2).mean())
    return R

import time
if __name__ == "__main__":
    # 第一批数据到达时的操作
    time_start = time.time()
    for i in range(1):
        data_dir = '../data/'
        # path_str:数据批次
        path_str = str(30)
        C_point_index = str(26)
        #print("------------------------------------------------第", path_str,
        #      "批数据------------------------------------------------------")
        data_dir_e = '../../data/KDDCUP_t/data_progressing/data_' + path_str + '_normal_minmax.csv'

        data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
        data_test = np.array(data_test)

        norm_data = data_test[:, 0:data_dim]



        dsc = DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   alpha=alpha_dsc, batch_num=path_str,
                                   C_point_index=C_point_index, model_path=model_path)

        DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)

    time_end = time.time()
    time_sum = time_end - time_start
    print("time_sum", time_sum)

