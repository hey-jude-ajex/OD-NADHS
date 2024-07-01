import numpy as np
import DSCModel
import DSC_t

import DSCModel_NChange


import pandas as pd
import SpectralCluster
import tensorflow as tf
import time
tf.random.set_seed(7)


data_dir = '../../data/'
# data_dir_e = '../data/test_data/test_normal_result_minmax_1.csv'
data_dir_e = '../../data/KDDCUP_t/data_progressing/data_36_normal_minmax.csv'
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

'''
def Judge_Change_point(norm_data, C_point_index):
    dsc = Judege_Change_point.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                               batch_size=batch_size,
                               alpha=alpha_dsc,
                               model_path=model_path)
    y_intput = Judege_Change_point.Judge_Change_point(dsc, model_path, norm_data, C_point_index)

    return y_intput
'''

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
    network_architecture = dict(n_hidden_enc_1=parameter['n_hidden_enc_1'][int(batch_num)-1],  # 1nd layer encode neurons
                                n_hidden_enc_2=parameter['n_hidden_enc_2'][int(batch_num)-1],  # 2nd layer encode neuron2
                                n_hidden_dec_1=parameter['n_hidden_dec_2'][int(batch_num)-1],  # 1nd layer decode neurons
                                n_hidden_dec_2=parameter['n_hidden_enc_1'][int(batch_num)-1],  # 2nd layer decode neurons
                                n_input=dim)  # dimension of data input
    alpha_dsc = {
        'a1':parameter['alpha_dsc_a1'][int(batch_num)-1],
        'a2':parameter['alpha_dsc_a2'][int(batch_num)-1],
    }
    # ro = 0.7
    # ro = max(0.4 - (5 - 1) / 10 * 0.1, 0.1)
    ro = parameter['ro'][int(batch_num)-1]
    d = parameter['d'][int(batch_num)-1]
    alpha = parameter['alpha'][int(batch_num)-1]


    # DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate, batch_size=batch_size,
    #                                    alpha=alpha_dsc, batch_num=batch_num, C_point_index = C_point_index,
    #                                    model_path=model_path)
    #print("Off57", type(batch_num), batch_num)
    #DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
    y_intput, coef = DSCModel_NChange.coef_get_NChange(model_path, norm_data, C_point_index, batch_num)
    #d = network_architecture['n_hidden_enc_2']
    time_s = time.time()
    # X_total_var = np.mean(np.var(coef, axis=0))
    # print(X_total_var.shape)
    # print("109", X_total_var)
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
        path_str = str(36)
        C_point_index = str(36)
        #print("------------------------------------------------第", path_str,
        #      "批数据------------------------------------------------------")
        data_dir_e = '../../data/KDDCUP_t/data_progressing/data_' + path_str + '_normal_minmax.csv'

        data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
        data_test = np.array(data_test)

        norm_data = data_test[:, 0:data_dim]

        # 判断到来的数据是否为Change_point

        # Y_test = Judge_Change_point(norm_data, C_point_index)
        # R = get_RMSE(norm_data, np.array(Y_test))
        # print("R", R)

        dsc = DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   alpha=alpha_dsc, batch_num=path_str,
                                   C_point_index=C_point_index, model_path=model_path)

        DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)

    time_end = time.time()
    time_sum = time_end - time_start
    print("time_sum", time_sum)
    #y_intput, coef = DSCModel_NChange.coef_get_NChange(dsc, model_path, norm_data)
    #d = network_architecture['n_hidden_enc_2']
    #C = SpectralCluster.thrC(coef, ro)
    # 相关性矩阵
    #grp, L = SpectralCluster.post_proC(C, K, d, alpha)

    import Change_point_train

    '''
    data_dir = '../data/'
    path_str = str(2 + i)
    print("------------------------------------------------第", path_str,
          "批数据------------------------------------------------------")
    data_dir_e = '../data/data_cut/change_point_data_progressing/data_' + path_str + '_normal_minmax.csv'
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
    data_test = np.array(data_test)
    print("124", data_test.shape)
    norm_data = data_test[:, 0:data_dim]

    dsc = DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                               batch_size=batch_size,
                               alpha=alpha_dsc, batch_num=path_str,
                               model_path=model_path)
    # Initial_parameter：初始化的模型参数
    # Initial_parameter = DSCModel_NChange.DSC.Initial_parameter_output(dsc)
    # print("134", type(Initial_parameter[0]))
    DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
    y_intput, coef = DSCModel_NChange.coef_get_NChange(dsc, model_path, norm_data)
    d = network_architecture['n_hidden_enc_2']
    C = SpectralCluster.thrC(coef, ro)
    # 相关性矩阵
    grp, L = SpectralCluster.post_proC(C, K, d, alpha)
    # 后面批次数据到达时的操作
    for i in range(3):
        tf.compat.v1.reset_default_graph()
        ata_dir = '../data/'
        path_str = str(3 + i)
        print("------------------------------------------------第", path_str,
              "批数据------------------------------------------------------")
        data_dir_e = '../data/data_cut/change_point_data_progressing/data_' + path_str + '_normal_minmax.csv'
        data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
        data_test = np.array(data_test)
        # norm_dat：去除原数据的最后一列标签
        norm_data = data_test[:, 0:data_dim]
        dsc = DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   alpha=alpha_dsc, batch_num=path_str,
                                   model_path=model_path)
        DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
        y_intput, coef = DSCModel_NChange.coef_get_NChange(dsc, model_path, norm_data)
        d = network_architecture['n_hidden_enc_2']
        C = SpectralCluster.thrC(coef, ro)
        # 相关性矩阵
        grp, L = SpectralCluster.post_proC(C, K, d, alpha)
    '''
    '''
    tf.compat.v1.reset_default_graph()
    data_dir_1 = '../data/'
    path_str = str(2)
    data_dir_e = '../data/data_cut/change_point_data_progressing/data_' + path_str + '_normal_minmax.csv'
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
    data_test = np.array(data_test)
    norm_data = data_test[:, 0:27]
    dsc = DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   alpha=alpha_dsc, batch_num=path_str,
                                   model_path=model_path)
    DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
    y_intput_1, coef = DSCModel_NChange.coef_get_NChange(dsc, model_path, norm_data)
    d = network_architecture['n_hidden_enc_2']
    C = SpectralCluster.thrC(coef, ro)
    # 相关性矩阵
    grp_1, L_1 = SpectralCluster.post_proC(C, K, d, alpha)
      

    tf.compat.v1.reset_default_graph()
    data_dir_ = '../data/'
    path_str = str(3)
    print("------------------------------------------------第", path_str,
          "批数据------------------------------------------------------")
    data_dir_e = '../data/data_cut/change_point_data_progressing/data_' + path_str + '_normal_minmax.csv'
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
    data_test = np.array(data_test)
    print("124", data_test.shape)
    norm_data = data_test[:, 0:27]

    dsc = DSCModel_NChange.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                               batch_size=batch_size,
                               alpha=alpha_dsc, batch_num=path_str,
                               model_path=model_path)
    # Initial_parameter：初始化的模型参数
    # Initial_parameter = DSCModel_NChange.DSC.Initial_parameter_output(dsc)
    # print("134", type(Initial_parameter[0]))
    DSCModel_NChange.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
    y_intput_, coef = DSCModel_NChange.coef_get_NChange(dsc, model_path, norm_data)
    d = network_architecture['n_hidden_enc_2']
    C = SpectralCluster.thrC(coef, ro)
    # 相关性矩阵
    grp_, L_ = SpectralCluster.post_proC(C, K, d, alpha)
    
    tf.compat.v1.reset_default_graph()
    data_dir_1 = '../data/'
    path_str = str(5)
    print("------------------------------------------------第", path_str,
          "批数据------------------------------------------------------")
    data_dir_e = '../data/data_cut/change_point_data_progressing/data_' + path_str + '_normal_minmax.csv'
    data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
    data_test = np.array(data_test)
    print("124", data_test.shape)
    norm_data = data_test[:, 0:27]

    dsc = DSC_t.DSC(network_architecture=network_architecture, learning_rate=learning_rate,
                               batch_size=batch_size,
                               alpha=alpha_dsc, batch_num=path_str,
                               model_path=model_path)
    # Initial_parameter：初始化的模型参数
    # Initial_parameter = DSCModel_NChange.DSC.Initial_parameter_output(dsc)
    # print("134", type(Initial_parameter[0]))
    DSC_t.train_model_NChange(dsc, norm_data, training_epochs=training_epochs_1)
    y_intput_1, coef = DSC_t.coef_get_NChange(dsc, model_path, norm_data)
    d = network_architecture['n_hidden_enc_2']
    C = SpectralCluster.thrC(coef, ro)
    # 相关性矩阵
    grp_1, L_1 = SpectralCluster.post_proC(C, K, d, alpha)
'''