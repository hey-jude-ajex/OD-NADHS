import time

import numpy as np
import pandas as pd
import DSCModel
#import utilss
from sklearn import cluster
from sklearn import decomposition
import matplotlib as plt
import pandas as pd
import Partial_training_online
import copy
import numpy as np



class Cluster_Detection(object):
    def __init__(self, Large_cluster_threshold, Similarity_threshold, Microcluster_threshold, norm_data, y_input, L, R_CP):
        self.Large_cluster_threshold = Large_cluster_threshold
        self.Similarity_threshold = Similarity_threshold
        self.Microcluster_threshold = Microcluster_threshold
        self.norm_data = norm_data
        self.y_input = y_input
        self.L = L
        self.R_CP = R_CP


        self.data_dim = self.norm_data.shape[1]-1

    def quicksort_desc_with_index(self, array):
        # 首先，我们使用 enumerate 函数来创建一个元组的列表，其中包含原始数组中每个元素的索引和值
        indexed_array = list(enumerate(array))

        # 然后，我们使用 sorted 函数进行排序。我们提供一个 lambda 函数作为 "key" 参数，
        # 这个函数将元组的第二个元素（即数组的值）作为排序的关键字。
        # 我们添加了一个 "-" 符号以实现降序排序
        sorted_array = sorted(indexed_array, key=lambda x: -x[1])

        # 最后，我们返回排序后的数组，但这次我们只返回值，而不返回索引
        sorted_values = [x[1] for x in sorted_array]
        original_indices = [x[0] for x in sorted_array]

        return sorted_values, original_indices

    def get_RMSE(self, y_int, y_out):
        R = np.sqrt(((y_int - y_out) ** 2).mean())
        return R

    def SLC(self):

        # 将每一行的元素相加,将矩阵压缩为一列
        c_sum = np.sum(self.L, axis=1)
        # indices 是快排前数据的原始位置下标
        sort, indices = self.quicksort_desc_with_index(c_sum)
        index_out = [i for i in range(self.L.shape[0])]
        #print(index_out)
        #对数据进行扫描
        #存放簇心的数组
        #indices = np.random.choice(np.array(index_out))
        center_index = []
        #存放聚出的簇
        C = []
        #j为簇的标签
        j = 0
        c_data = self.norm_data
        while len(index_out) != 0:
            #初始聚簇
            if len(C) == 0:
                # 确定簇心下标
                center_index.append(indices[0])
                C.append([])
                C[j].append(indices[0])
                index_out.remove(indices[0])

            else:
                # c_i为当前要处理的数据
                c_i = index_out[0]
                similarity_max = 0
                for index, e in enumerate(center_index):
                    if self.L[c_i][e] > similarity_max:
                        similarity_max = self.L[c_i][e]
                        self.i = index
                if similarity_max > self.Similarity_threshold:
                    #相似度大于阈值，划归到对应的簇中
                    C[self.i].append(c_i)
                    index_out.pop(0)
                else:
                    #print("102", c_i)
                    center_index.append(c_i)
                    j = j + 1
                    C.append([])

                    C[j].append(c_i)
                    index_out.pop(0)
        # print("110", C[3])
        # print('108', len(C), len(center_index))
        # for index, c in enumerate(C):
        #
        #     print(index, c)
        return C


    def Detection(self, C, data_index):
        shape = (1, self.norm_data.shape[1])  # 指定形状为 (3, 4)
        array = np.zeros(shape)
        #存放初筛结果
        #contextual
        A_con  = []
        #collection
        A_col = []
        #point
        A_poi = []

        m = 0
        L_S = []


        #存放细筛结果：
        Contextual_anomalies = []
        Collective_anomalies = np.zeros(shape)
        Point_anomalies = np.zeros(shape)

        #存放初筛得到的正常数据
        pre_Y  =[]
        pre_X  =[]

        Similarity_all = []

        for i, c in enumerate(C):
            C_A = np.array(self.norm_data)[c]
            #每个簇对应的输出数据
            #print("132", self.Large_cluster_threshold)
            if len(c) > self.Microcluster_threshold:
                #print(C_A[:, self.data_dim], len(C_A))
                #处理大簇
                #print(c[0], type(c))
                for _, con in enumerate(c, start=0):
                    if self.L[c[0]][con] < self.Large_cluster_threshold:
                        L_S.append((con, self.L[c[0]][con]))
                        #存放上下文异常的初筛结果
                        A_con.append(con)

                    else:
                        pre_X.append(self.norm_data[con][0:self.data_dim])
                        pre_Y.append(self.y_input[con])

            elif 1< len(c) <= self.Microcluster_threshold:
                #A_col.append([])
                A_col.append(np.array(c))
                #A_col = np.vstack((A_col, C_A))
                #Fine-tunning
                # if R_o > self.R_threshold:
                #     Collective_anomalies = np.vstack((Collective_anomalies, C_A))
            elif len(c) ==1 :
                A_poi.append(c)
                #A_poi = np.vstack((A_poi, self.norm_data[c]))
                #Fine-tunning
                # if R_o > self.R_threshold:
                #     Point_anomalies = np.vstack((Point_anomalies, self.norm_data[c]))
            m += 1
            #去掉A_poi第一行的0
            print(C_A[:, self.data_dim], len(C_A))





        #存放r阈值-时间


        #self.R_threshold = 0.32764872576169213
        #self.R_threshold = 0.281872800295018  #12
        #self.R_threshold = 0.33482351537691163  #22
        #self.R_threshold = 0.31512234028511593 #28
        #self.R_threshold = 0.2539636302010785 #36

        #36:2
        if len(A_con) != 0:
            self.R_threshold = self.get_RMSE(np.array(pre_X), pre_Y)*1.4
            # print("-----------------------------------", data_index)
            # print(len(self.R_CP), int(data_index), self.R_CP)
            if len(self.R_CP) < int(data_index):
                self.R_CP.append(self.R_threshold)
                #print("176", self.R_CP)

        else:
            if len(self.R_CP)<int(data_index):
                self.R_threshold = self.R_CP[int(data_index)-2]
                self.R_CP.append(self.R_CP[int(data_index)-2])

        #self.R_threshold = 0.31
        #self.R_threshold = 0.31
        #print("self.R_threshold", self.R_threshold)
        Similarity_all = []
        L_S = np.array(L_S)
        #细筛
        for i,con  in enumerate(A_con):
            if self.get_RMSE(self.norm_data[con][0:self.data_dim], np.array(self.y_input)[con]) > self.R_threshold:
                Contextual_anomalies.append(self.norm_data[con])
                Similarity_all.append(L_S[i])
        #-----------------------------------------------------------------------------

        if len(Similarity_all) != 0:
            Similarity_all = np.array(Similarity_all)
            #print("--------------------------------------------------------")
            sorted_array = Similarity_all[Similarity_all[:, 1].argsort()]
            # 输出最小的10行
            result = sorted_array[:30, :]
            # print("result", len(result))
            # print("result", result)
            # print("--------------------------------------------------------")
        #------------------------------------------------------------------------------
        for _, col in enumerate(A_col):
            #print("171", self.norm_data[col][:, self.data_dim], self.get_RMSE(self.norm_data[col][:, 0:self.data_dim], np.array(self.y_input)[col]))

            if self.get_RMSE(self.norm_data[col][:, 0:self.data_dim], np.array(self.y_input)[col]) > self.R_threshold:
                Collective_anomalies = np.vstack((Collective_anomalies, self.norm_data[col]))

        for p in A_poi:
            #print("176", np.array(self.norm_data[p]).shape)
            if self.get_RMSE(self.norm_data[p][:, 0:self.data_dim], np.array(self.y_input)[p]) > self.R_threshold:
                Point_anomalies = np.vstack((Point_anomalies, self.norm_data[p]))


        # A_col = A_col[1:]
        # A_con = np.array(A_con)
        # A_poi = A_poi[1:]

        Collective_anomalies = Collective_anomalies[1:]
        #Contextual_anomalies = np.array(Contextual_anomalies)
        #print("192", len(Contextual_anomalies))
        Point_anomalies = Point_anomalies[1:]
        # if len(Contextual_anomalies) != 0:
        #     #Similarity_all = np.array(Similarity_all)
        #     print("--------------------------------------------------------")
        #     #sorted_array = Similarity_all[Similarity_all[:, 1].argsort()]
        #     # 输出最小的10行
        #     #result = sorted_array[:20, :]
        #     #print("185", result)
        #     print("Contextual_anomalies", len(np.array(Contextual_anomalies)[:, self.data_dim]), np.array(Contextual_anomalies)[:, self.data_dim])
        #     # print("--------------------------------------------------------")
        Contextual_anomalies = np.array(Contextual_anomalies)
        # print("Collective_anomalies", Collective_anomalies[:, self.data_dim])
        # print("Point_anomalies", Point_anomalies[:, self.data_dim])


        labels_outliers = []  #存放筛选出的数据的标签
        labels_outliers = np.concatenate((Collective_anomalies[:, self.data_dim], Point_anomalies[:, self.data_dim]), axis=0)
        if len(Contextual_anomalies)!=0:
            labels_outliers = np.concatenate((labels_outliers, Contextual_anomalies[:, self.data_dim]), axis=0)
        print(labels_outliers)

        return labels_outliers

