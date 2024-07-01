

import time
import tensorflow as tf
import numpy as np
import Exception_handling
import pandas as pd
import Partial_training_online
import warnings
import Change_point_parameter
import cluster_detection
import json
import change_point_detection
warnings.filterwarnings("ignore")


#if __name__ == "__main__":

data_dir_test = '../../data/stock_t/time-series-toMinmax/a_1_minmax.csv'
data_test = pd.read_csv(data_dir_test, header=None, dtype=float)
data_test = np.array(data_test)
#d = 1.1e-08

def F1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def get_RMSE( y_int, y_out):
    R = np.sqrt(((y_int - y_out) ** 2).mean())
    return R
import numpy

if __name__ == "__main__":

    df = pd.read_json('./parameters_dict.json', orient='records', lines=True)

    # 将 DataFrame 转换为字典列表
    parameters_all = df.to_dict(orient='records')

    # 格式化并保存为易读的 JSON 文件
    with open('./parameters_formatted.json', 'w', encoding='utf-8') as f:
        json.dump(parameters_all, f, ensure_ascii=False, indent=4)
    data_batchs = 0
    time_x = []
    time_all = 0
    precision = []
    recall = []
    outliers_fraction = []

    n_outliers_all = 0
    n_corrected_outliers_all = 0
    #分别存放召回的各类异常的总数
    n_isop_all = 0
    n_Lclu_al = 0
    n_Gclu_all = 0
    #分别存放筛选出的真正小簇异常点数以及筛选出的小簇异常点总数
    Correct_microcluster_outliers_all = 0
    Micclustered_data_all = 0
    #分别存放筛选出的真正的孤立点异常数以及筛选出的孤立点异常总数
    Correct_Orphaned_outliers_all = 0
    Orphaned_data_all = 0
    # 存放数据预处理后真正的大簇异常数量
    Great_clusters_outliers_quantity = 0
    #存放筛选出的大簇异常数量
    G_clusters_outliers_all = 0
    #参数设定
    # 存放微簇和孤立点异常的下标
    Mic_index_all = []
    i_index_all = []
    G_index_all = []

    R_CP = []
    time_start = time.time()  # 线上检测分开始时间
    for i in range(20):
        path_str = str(i+1)
        parameters = parameters_all[int(path_str) - 1]
        print("86", parameters)
        print("88", parameters['C_point_index'], type(parameters['C_point_index']))
        C_point_index = str(parameters['C_point_index'])
        


        Similarity_threshold = 0.65
        Large_cluster_threshold = 0.7
        Microcluster_threshold = 11


        data_path = '../../data/'
        data_dir = '../../data/stock_t/time-series-toMinmax/a_'+ path_str+ '_minmax.csv'

        data = pd.read_csv(data_dir, header=None, dtype=float)
        data = np.array(data)
        print("shape", data.shape)

        #总离群点
        out_count = data.shape[0] * 0.1
        #孤立点个数
        out_count_isop = data.shape[0]*0.02
        #小簇离群点
        out_count_Lclu = data.shape[0] * 0.06
        #大簇内部低相似度点
        out_count_Gclu = data.shape[0] * 0.02
        norm_data = data[:, 0:data.shape[1]-1]                                    #normdata是去掉标签的数据 [500, 41]
        RMSE_CP_D = 0.015

        Change_D = change_point_detection.Change_point(norm_data, RMSE_CP_D, path_str, parameters)
        Y, L = Change_D.judgment()


        R_pre = get_RMSE(norm_data, np.array(Y))
        print("R_pre", R_pre)
        #R_threshold = 0.16
        #print(R_threshold)

        C_D = cluster_detection.Cluster_Detection(Large_cluster_threshold=Large_cluster_threshold,
                                                  Similarity_threshold=Similarity_threshold,
                                                  Microcluster_threshold=Microcluster_threshold,
                                                  norm_data=data, y_input=Y, L=L, R_CP = R_CP)

        C = C_D.SLC()
        labels_outliers = C_D.Detection(C, path_str)
        R_pre = C_D.get_RMSE(norm_data, np.array(Y))


        # 准确率，正确率计算
        n_corrected_outliers=0
        n_isop=0
        n_Lclu=0
        n_Gclu = 0

#result
        #统计筛选出的各类异常的总数
        for k in labels_outliers:
            if k < 0:
                n_corrected_outliers = n_corrected_outliers + 1
                if k == -2:
                    n_isop = n_isop+1
                elif k == -3:
                    n_Gclu = n_Gclu+1
                else: n_Lclu = n_Lclu+1

        n_Lclu_al = n_Lclu_al + n_Lclu
        n_isop_all = n_isop_all+n_isop

        n_corrected_outliers_all = n_corrected_outliers_all + n_corrected_outliers
        n_outliers_all = n_outliers_all + len(labels_outliers)
        if parameters['Gco_Q'] != 0:
            n_Gclu_all = n_Gclu_all + n_Gclu

            Great_clusters_outliers_quantity = Great_clusters_outliers_quantity + parameters['Gco_Q']
        data_batchs = data_batchs+1

    time_end = time.time()
    time_sum = time_end - time_start
    print("time_sum", time_sum)
    print("147", n_isop_all, data_batchs*data_test.shape[0]*0.02)
    print("运行时间", time_all)
    print("G_Q:", Great_clusters_outliers_quantity,
          (data_batchs * data_test.shape[0] * 0.08 + Great_clusters_outliers_quantity))

    print("all_recall", (n_isop_all + n_Lclu_al + n_Gclu_all) / (data_batchs*data_test.shape[0]*0.08 + Great_clusters_outliers_quantity))
    print((data_batchs*data_test.shape[0]*0.08 + Great_clusters_outliers_quantity))
    print("i_accuracy", n_isop_all / (data_batchs * data_test.shape[0] * 0.02))
    print("n_Lclu_al", n_Lclu_al, data_batchs * data_test.shape[0] * 0.06)
    print("L_accracy", n_Lclu_al / (data_batchs * data_test.shape[0] * 0.06))
    print("n_Gclu_all", n_Gclu_all, Great_clusters_outliers_quantity)
    if n_Gclu_all != 0:
        #n_Gclu_all = 0
        print("G_accracy", n_Gclu_all / Great_clusters_outliers_quantity)
    print("--------------------------------------------------------------------------")
    print("n_corrected_outliers_all", n_corrected_outliers_all, n_Gclu_all)
    print("Tatal_precision", n_corrected_outliers_all / n_outliers_all)


    print("--------------------------------------------------------------------------62%")
    print("F1", F1_score(n_corrected_outliers_all / n_outliers_all, (n_isop_all + n_Lclu_al + n_Gclu_all) / (
            data_batchs * data_test.shape[0] * 0.08 + Great_clusters_outliers_quantity)))


#画图
'''

    import  matplotlib.pyplot as plt
    x_ticks = time_x
    #print(x_ticks.shape)
    print("160", precision)
    print(recall)
    plt.plot(x_ticks, precision, 'b*--', alpha= 0.5,  label = 'acc', linewidth = 3.0)
    plt.plot(x_ticks, recall, 'rs--', alpha=0.5, label='recall', linewidth=3.0)
    for a, b in zip(x_ticks, precision):
        plt.text(a, b, str(b), ha = 'center', va='bottom', fontsize = 8)
    for a, b in zip(x_ticks, recall):
        plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)
    plt.legend()
    #plt.plot(x, acc, color = '#FF0000', label = 'acc', linewidth = 3.0)
    plt.xlabel('outliers_fraction')
    plt.ylabel('rate')
    plt.show()

'''















