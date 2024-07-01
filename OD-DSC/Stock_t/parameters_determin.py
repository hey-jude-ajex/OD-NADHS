import time

import numpy as np
import Exception_handling
import pandas as pd
import Partial_training_online
import warnings
warnings.filterwarnings("ignore")


#if __name__ == "__main__":


#d = 1.1e-08

def F1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

import numpy

if __name__ == "__main__":
    time_x = []
    precision = []
    recall = []
    outliers_fraction = []


    n_isop_all = 0
    n_Lclu_al = 0
    n_Gclu_all = 0
    precision_P_all = 0
    precision_PN_all = 0
    outliers_all = 0
#异常点的比例：10%
    for i in range(1):
        print("28", i)
        #time.sleep(10)
        print("------------------------------------------------第", i+1, "批数据------------------------------------------------------")
        print("time：", time.time())
        data_dir = '../data/'
        path_str = str(i +5)
        data_dir_e = '../data/data_cut/change_point_data_progressing/data_'+ path_str+ '_normal_minmax.csv'
        data_test = pd.read_csv(data_dir_e, header=None, dtype=float)
        data_test = np.array(data_test)

        #总异常点
        out_count = data_test.shape[0] * 0.1
        #孤立点个数
        out_count_isop = data_test.shape[0]*0.02
        #小簇异常点
        out_count_Lclu = data_test.shape[0] * 0.06
        #大簇内部大于大簇阈值的异常点
        out_count_Gclu = data_test.shape[0] * 0.02
        norm_data = data_test[:, 0:27]                                    #normdata是去掉标签的数据 [500, 41]



        Y, L, grp = Offline_part.get_coef_NChange_test(norm_data, path_str)
        print(L)
        #numpy.savetxt('../data/data_cut/coef.csv', L, delimiter=',')
        print("---------------------------grp---------------------------")
        print(grp, len(grp))
        clusters_corrects, clusters_y_corrects, clusters_outliers, clusters_y_outliers, clusters_outliers_Gclu = Exception_handling.cluster_outlier(L, data_test, Y)
        labels_outliers = Exception_handling.Great_clusters_outliers(clusters_outliers_Gclu)
        R = Exception_handling.get_RMSE(clusters_corrects, clusters_y_corrects)
        print(R)
        print("----------------------------------离群点微簇RMSE----------------------------------------")
        clusters_corrects_new, clusters_y_corrects_new, labels_outliers, precision_P, precision_PN= Exception_handling.Meticulous_screening(clusters_outliers, clusters_y_outliers, clusters_corrects, clusters_y_corrects, R, labels_outliers=labels_outliers)
        R_new = Exception_handling.get_RMSE(clusters_corrects_new, clusters_y_corrects_new)
        precision_P_all = precision_P_all + precision_P
        precision_PN_all = precision_PN_all + precision_PN
        print(labels_outliers)
        print("-----------------------------操作前后RMSE----------------------------------------------")
        print(R, R_new)
        # -----------------------------------------------------------------------
        # 准确率，正确率计算
        n=0
        n_isop=0
        n_Lclu=0
        n_Gclu = 0

#result
        for k in labels_outliers:
            if k < 0:
                n = n + 1
                if k == -3:
                    n_isop = n_isop+1
                elif k == -4:
                    n_Gclu = n_Gclu+1
                else: n_Lclu = n_Lclu+1
        # 精准率
        p = n / len(labels_outliers)
        p = round(p, 2)
        precision.append(p)
        # 召回率
        r = n / out_count
        r = round(r, 2)
        #孤立点
        print("102", out_count_isop)
        r_isop = n_isop/ out_count_isop
        r_isop = round(r_isop, 2)
        print("筛选出的异常点总数：", len(labels_outliers))
        print("孤立点召回率：", r_isop)
        print("孤立点总数", n_isop)
        #小簇
        r_Lclu = n_Lclu / out_count_Lclu
        r_Lclu = round(r_Lclu, 2)
        print("小簇异常召回率：", r_Lclu)
        print("小簇总数：", n_Lclu)
        #大簇
        print("112", out_count_Lclu)
        r_Gclu = n_Gclu / out_count_Gclu
        r_Gclu = round(r_Gclu, 2)
        print("大簇低相似度异常召回率：", r_Gclu)
        print("大簇异常总数：", n_Gclu)
        n_Lclu_al = n_Lclu_al + n_Lclu
        n_isop_all = n_isop_all+n_isop
        n_Gclu_all = n_Gclu_all+n_Gclu
        outliers_all = outliers_all + len(labels_outliers)
        #F1_score = F1_score(p, r)
        #print("F1_score:", F1_score)
        recall.append(r)
        x = data_test.shape[0] * 0.1
        # f = lambda x: '%.4f' % x
        f = lambda x: '%.2f%%' % (x * 100)
        print("146", type(f))
        outliers_fraction.append(x)
    print("147", outliers_all)
    P = precision_P_all/precision_PN_all
    print("precision", precision_P_all/precision_PN_all)
    print("acc", (n_isop_all+n_Lclu_al+n_Gclu_all)/outliers_all)
    print("i_recall", n_isop_all/50)
    print("L_recall", n_Lclu_al/150)
    print("G_recall", n_Gclu_all/50)
    print("all_recall", (n_isop_all + n_Lclu_al+n_Gclu_all)/250)
    print("F1", F1_score(P, n_Lclu_al/150))

