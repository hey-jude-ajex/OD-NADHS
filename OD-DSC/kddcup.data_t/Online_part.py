

import time
import tensorflow as tf
import numpy as np

import pandas as pd
import Partial_training_online
import warnings
import Change_point_parameter
import cluster_detection
import json
import change_point_detection

warnings.filterwarnings("ignore")
tf.random.set_seed(7)

#if __name__ == "__main__":

data_dir_test = '../../data/KDDCUP_t/data_progressing/data_1_normal_minmax.csv'
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
    #存放数据预处理后真正的大簇异常数量
    Great_clusters_outliers_quantity = 0
    #存放筛选出的大簇异常数量
    G_clusters_outliers_all = 0


    #存放微簇和孤立点异常的下标

    TN = 0
    FP = 0
    FN = 0

    R_CP = []
    time_start = time.time()  # 线上检测分开始时间
    for i in range(30):
        path_str = str(i+1)
        parameters = parameters_all[int(path_str)-1]
        C_point_index = str(parameters['C_point_index'])


        # 参数设定
        Similarity_threshold = 0.66
        Large_cluster_threshold = 0.75
        Microcluster_threshold = 19
        # R_threshold:细筛部分用到的RMSE阈值
        R_threshold = parameters['R_threshold']



        data_path = '../../data/'
        #加载数据
        data_dir = '../../data/KDDCUP_t/data_progressing/data_'+ path_str+ '_normal_minmax.csv'

        data = pd.read_csv(data_dir, header=None, dtype=float)
        data = np.array(data)
        #总离群点
        print("shape", data.shape)
        out_count = data.shape[0] * 0.1
        #孤立点个数
        out_count_isop = data.shape[0]*0.02
        #小簇离群点
        out_count_Lclu = data.shape[0] * 0.06
        #大簇内部低相似度点
        out_count_Gclu = data.shape[0] * 0.02

        print(data.shape, type(data))
        norm_data = data[:, 0:27]                                    #normdata是去掉标签的数据 [500, 41]

        RMSE_CP_D = 12
        Change_D = change_point_detection.Change_point(norm_data, RMSE_CP_D, path_str, parameters)
        Y, L = Change_D.judgment()
        #Y, L= Partial_training_online.get_coef_NChange_test(norm_data, path_str, C_point_index=C_point_index, parameter=parameters)
        R_pre = get_RMSE(norm_data, np.array(Y))


        C_D = cluster_detection.Cluster_Detection(Large_cluster_threshold=Large_cluster_threshold,
                                                         Similarity_threshold=Similarity_threshold, Microcluster_threshold=Microcluster_threshold,
                                                          norm_data=data, y_input=Y, L=L, R_CP = R_CP)
        time_start_1 = time.time()
        C = C_D.SLC()


        labels_outliers = C_D.Detection(C, path_str)
        time_end_1 = time.time()




        n_corrected_outliers=0
        n_isop=0
        n_Lclu=0
        n_Gclu = 0

        #统计筛选出的各类异常的总数
        for k in labels_outliers:
            if k < 0:
                n_corrected_outliers = n_corrected_outliers + 1
                if k == -3:
                    n_isop = n_isop+1
                elif k == -4:
                    n_Gclu = n_Gclu+1
                else: n_Lclu = n_Lclu+1
        n_Lclu_al = n_Lclu_al + n_Lclu
        n_isop_all = n_isop_all+n_isop

        n_corrected_outliers_all = n_corrected_outliers_all + n_corrected_outliers
        n_outliers_all = n_outliers_all + len(labels_outliers)

        if parameters['Gco_Q'] != 0:
            n_Gclu_all = n_Gclu_all + n_Gclu

            Great_clusters_outliers_quantity = Great_clusters_outliers_quantity + parameters['Gco_Q']
           # print("133", parameters['Gco_Q'][int(path_str)-1])
        data_batchs = data_batchs+1


        index = data[:, data.shape[1] - 1]
        index = list(index)
        labels_o = list(labels_outliers)

    time_end = time.time()
    time_sum = time_end - time_start
    print("time_sum", time_sum)
    #print("147", n_isop_all, data_batchs*data_test.shape[0]*0.02)
    print("运行时间", time_all)

    #---------------------------------------------------------
    TP = n_corrected_outliers_all
    #acc  = (TP + TN)/(TP + TN + FP + FN)

    print("G_Q:", Great_clusters_outliers_quantity, (data_batchs*data_test.shape[0]*0.08 + Great_clusters_outliers_quantity))

    print("i_accuracy", n_isop_all/(data_batchs*data_test.shape[0]*0.02))
    print("L_accracy", n_Lclu_al/(data_batchs*data_test.shape[0]*0.06))
    print("n_Gclu_all", n_Gclu_all, Great_clusters_outliers_quantity)
    if n_Gclu_all!= 0:
        print("G_accracy", n_Gclu_all / Great_clusters_outliers_quantity)
    print(n_corrected_outliers_all, n_outliers_all)
    print("--------------------------------------------------------------------------")
    #print("Accuracy", acc)
    print("Tatal_precision", n_corrected_outliers_all / n_outliers_all)
    print("all_recall", (n_isop_all + n_Lclu_al + n_Gclu_all) / (
                data_batchs * data_test.shape[0] * 0.08 + Great_clusters_outliers_quantity))
    print("F1", F1_score(n_corrected_outliers_all / n_outliers_all, (n_isop_all + n_Lclu_al + n_Gclu_all) / (
                data_batchs * data_test.shape[0] * 0.08 + Great_clusters_outliers_quantity)))


    print(Correct_Orphaned_outliers_all, Orphaned_data_all, Correct_microcluster_outliers_all, n_corrected_outliers_all, n_outliers_all)

















