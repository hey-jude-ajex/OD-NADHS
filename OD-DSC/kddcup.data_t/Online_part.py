

import time
import tensorflow as tf
import numpy as np
import Exception_handling
import pandas as pd
import Partial_training_online
import warnings
import Change_point_parameter
import cluster_detection
warnings.filterwarnings("ignore")
tf.random.set_seed(7)

#if __name__ == "__main__":

data_dir_test = '../../data/KDDCUP_t/data_progressing/data_35_normal_minmax.csv'
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
    #data_batchs:change_point点到来之前到达的数据的批次
    #parameters:离群点所用的参数
    parameters = Change_point_parameter.parameters
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
    #print("线上部分开始时间", time_start)
    for i in range(1):
        path_str = str(i+1)
        C_point_index = str(parameters['C_point_index'][int(path_str)-1])

        # 参数设定
        Similarity_threshold = 0.66
        Large_cluster_threshold = 0.75
        Microcluster_threshold = 19
        # R_threshold:细筛部分用到的RMSE阈值
        R_threshold = parameters['R_threshold'][int(path_str)-1]



        data_path = '../../data/'
        #加载数据
        data_dir = '../../data/KDDCUP_t/data_progressing/data_'+ path_str+ '_normal_minmax.csv'

        data = pd.read_csv(data_dir, header=None, dtype=float)
        #异常率为10%
        #总离群点
        print("shape", data.shape)
        out_count = data.shape[0] * 0.1
        #孤立点个数
        out_count_isop = data.shape[0]*0.02
        #小簇离群点
        out_count_Lclu = data.shape[0] * 0.06
        #大簇内部低相似度点
        out_count_Gclu = data.shape[0] * 0.02
        norm_data = data[:, 0:data.shape[1]-1]                                    #normdata是去掉标签的数据 [500, 41]

        Y, L= Partial_training_online.get_coef_NChange_test(norm_data, path_str, C_point_index=C_point_index, parameter=parameters)
        R_pre = get_RMSE(norm_data, np.array(Y))
        #print("R_pre", R_pre)
        #R_threshold = 0.37


        C_D = cluster_detection.Cluster_Detection(Large_cluster_threshold=Large_cluster_threshold,
                                                         Similarity_threshold=Similarity_threshold, Microcluster_threshold=Microcluster_threshold,
                                                          norm_data=data, y_input=Y, L=L, R_CP = R_CP)
        time_start_1 = time.time()
        C = C_D.SLC()


        labels_outliers = C_D.Detection(C, path_str)
        time_end_1 = time.time()

        print("time_CD", time_end_1-time_start_1)



        n_corrected_outliers=0
        n_isop=0
        n_Lclu=0
        n_Gclu = 0

#result
        #统计筛选出的各类异常的总数
        for k in labels_outliers:
            if k < 0:
                n_corrected_outliers = n_corrected_outliers + 1
                if k == -3:
                    n_isop = n_isop+1
                elif k == -4:
                    n_Gclu = n_Gclu+1
                else: n_Lclu = n_Lclu+1

        #print("大簇异常召回率：", r_Gclu)
        n_Lclu_al = n_Lclu_al + n_Lclu
        n_isop_all = n_isop_all+n_isop

        n_corrected_outliers_all = n_corrected_outliers_all + n_corrected_outliers
        n_outliers_all = n_outliers_all + len(labels_outliers)

        if parameters['Gco_Q'][int(path_str)-1] != 0:
            n_Gclu_all = n_Gclu_all + n_Gclu

            Great_clusters_outliers_quantity = Great_clusters_outliers_quantity + parameters['Gco_Q'][int(path_str)-1]
           # print("133", parameters['Gco_Q'][int(path_str)-1])
        data_batchs = data_batchs+1


        index = data[:, data.shape[1] - 1]
        index = list(index)
        labels_o = list(labels_outliers)
        #print("147", type(labels_o), len(labels_o))
        #print("150", type(index), len(index))
        # for i in labels_o:
        #     if i in index:
        #         index.remove(i)
        #
        # print(index, len(index))
        # for _,i in enumerate(index):
        #     if i >= 0:
        #         TN += 1
        #     if i < 0:
        #         FN += 1
        #
        # for _, i in enumerate(labels_o):
        #     if i >= 0:
        #         FP += 1






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

















