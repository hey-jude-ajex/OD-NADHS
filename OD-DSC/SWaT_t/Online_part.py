

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

data_dir_test = '../../data/SWaT_t/data_progressing/SWaT_1_minmax.csv'
data_test = pd.read_csv(data_dir_test, header=None, dtype=float)
data_test = np.array(data_test)
#d = 1.1e-08

def F1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def get_RMSE( y_int, y_out):
    R = np.sqrt(((y_int - y_out) ** 2).mean())
    return R

if __name__ == "__main__":
    #data_batchs:change_point点到来之前到达的数据的批次
    df = pd.read_json('./parameters_dict.json', orient='records', lines=True)

    # 将 DataFrame 转换为字典列表
    parameters_all = df.to_dict(orient='records')

    # 格式化并保存为易读的 JSON 文件
    with open('./parameters_formatted.json', 'w', encoding='utf-8') as f:
        json.dump(parameters_all, f, ensure_ascii=False, indent=4)


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
    for i in range(40):
        path_str = str(i+1)

        parameters = parameters_all[int(path_str) - 1]
        C_point_index = str(parameters['C_point_index'])

        Similarity_threshold = 0.5
        Large_cluster_threshold = 0.55
        Microcluster_threshold = 9
        # R_threshold:细筛部分用到的RMSE阈值
        R_threshold = parameters['R_threshold']

        print("------------------------------------------------第", i+1, "批数据------------------------------------------------------")
        data_path = '../../data/'
        data_dir = '../../data/SWaT_t/data_progressing/SWaT_'+ path_str+ '_minmax.csv'

        data = pd.read_csv(data_dir, header=None, dtype=float)
        data = np.array(data)
        out_count = 192
        #孤立点个数
        out_count_isop = data.shape[0]*0.02
        #小簇离群点
        out_count_Lclu = 192
        #大簇内部低相似度点
        out_count_Gclu = data.shape[0] * 0.02
        norm_data = data[:, 0:data.shape[1]-1]                                    #normdata是去掉标签的数据 [500, 41]

        RMSE_CP_D = 0.05
        Change_D = change_point_detection.Change_point(norm_data, RMSE_CP_D, path_str, parameters)
        Y, L = Change_D.judgment()
        R_pre = get_RMSE(norm_data, np.array(Y))


        C_D = cluster_detection.Cluster_Detection(Large_cluster_threshold=Large_cluster_threshold,
                                                  Similarity_threshold=Similarity_threshold,
                                                  Microcluster_threshold=Microcluster_threshold,
                                                  norm_data=data, y_input=Y, L=L, R_CP=R_CP)
        time_start_1  = time.time()
        C = C_D.SLC()
        labels_outliers = C_D.Detection(C, path_str)
        time_end_1 = time.time()

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



        #召回率



        r_Gclu = n_Gclu / out_count_Gclu
        r_Gclu = round(r_Gclu, 2)

        n_Lclu_al = n_Lclu_al + n_Lclu
        n_isop_all = n_isop_all+n_isop



        n_outliers_all = n_outliers_all + len(labels_outliers)
        n_corrected_outliers_all = n_corrected_outliers_all+n_corrected_outliers




        data_batchs = data_batchs+1

    time_end = time.time()
    time_sum = time_end - time_start


    print("all_recall", (n_isop_all + n_Lclu_al + n_Gclu_all) /(data_batchs*data_test.shape[0]*0.1))


    print("n_corrected_outliers_all", n_corrected_outliers_all, n_Gclu_all)


    print("Tatal_precision", n_corrected_outliers_all / n_outliers_all)
    print("n_outliers_all", n_outliers_all)
    print("--------------------------------------------------------------------------62%")
    print("F1", F1_score(n_corrected_outliers_all / n_outliers_all, (n_isop_all + n_Lclu_al + n_Gclu_all) /(data_batchs*data_test.shape[0]*0.1)))

