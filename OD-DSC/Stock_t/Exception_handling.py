import time

import numpy as np
import pandas as pd
import DSCModel
#import utilss
#from sklearn import cluster
from sklearn import decomposition
#import matplotlib as plt
#import pandas as pd
#import Partial_training_online
import copy



class Exception_handing(object):
    def __init__(self, Large_cluster_threshold, Similarity_threshold, Microcluster_threshold, R_threshold, norm_data, y_input, L):
        self.Large_cluster_threshold = Large_cluster_threshold
        self.Similarity_threshold = Similarity_threshold
        self.Microcluster_threshold = Microcluster_threshold
        self.R_threshold = R_threshold
        self.norm_data = norm_data
        self.y_input = y_input
        self.L = L
        self.data_dim = self.norm_data.shape[1]-1
        #print("24###########", self.data_dim)
    def get_S_index(self, Similarity_all, n):

        m = Similarity_all
        t = copy.deepcopy(m)

        min_number = []
        min_index = []
        for _ in range(n):
            number = min(t)
            index = t.index(number)
            t[index] = float('inf')
            min_number.append(number)
            min_index.append(index)
        t = []
        # print(min_number)
        # print(min_index)
        return min_index


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

    def cluster_outlier(self):
        # norm_data与y_intpuut相对应
        # 聚簇测试
        n = 0  # 簇标签
        clusters_corrects = np.ones((1, self.norm_data.shape[1] - 1))  # 存放正常数据点
        microclusters_collection = []  # 存放初筛出的微簇集合
        clusters_outliers_Gclu = np.ones((1, self.norm_data.shape[1]))  # 存放小于大簇阈值的异常点
        clusters_y_corrects = np.ones((1, self.norm_data.shape[1] - 1))  # 存放正常数据输出的数据
        clusters_y_outliers = []  # 存放筛选离群点的输出数据
        # c_data = norm_data
        c_data = self.norm_data  # [500, 42]
        Similarity_all = []
        Similarity_index_all = []
        # while norm_data.any():
        n = n + 1

        # 将每一行的元素相加,将矩阵压缩为一列
        c_sum = np.sum(self.L, axis=1)
        # indices 是快排前数据的原始位置下标
        sort, indices = self.quicksort_desc_with_index(c_sum)
        index_out = []
        for i in range(self.L.shape[0]):
            index_out.append(i)
        c = np.zeros(self.L.shape[0])
        while len(index_out) != 0:
            center_index = indices[0]
            #center_index = [0,10,20,30,40,50,60,70,80,90,100]
            # print("center_index", center_index)
            index = []
            mic_clusters = []
            mic_clusters_y = []
            clusters_labels = []
            clusters_labels_y = []
            clusters_t = []
            clusters_t_y = []
            clusters_center = []
            Similarity = []
            Similarity_index = []
            # 簇内数据点计数
            n = 0
            n_1 = 0
            for i in index_out:
                # print("114", i)

                if self.L[center_index, i] > self.Similarity_threshold and self.L[center_index, i] < self.Large_cluster_threshold:
                    clusters_center.append(i)
                    index_out.remove(i)
                    #
                    indices.remove(i)
                    Similarity.append(self.L[center_index, i])
                    Similarity_index.append(i)
                    mic_clusters.append(c_data[i])
                    mic_clusters_y.append(self.y_input[i])
                    clusters_labels.append(c_data[i])
                    clusters_labels_y.append(self.y_input[i])
                    # print("########", i, L[center_index, i])
                    self.L[i] = c
                    n = n + 1
                    n_1 = n_1 + 1
                #大簇
                elif self.L[center_index, i] >= self.Large_cluster_threshold:
                    mic_clusters.append(c_data[i])
                    mic_clusters_y.append(self.y_input[i])
                    clusters_t.append(c_data[i])
                    clusters_t_y.append(self.y_input[i])
                    clusters_center.append(i)
                    self.L[i] = c
                    index_out.remove(i)
                    # print("125", len(index_out))
                    indices.remove(i)

                    index.append(i)
                    n = n + 1

            # print("----", n, "----", clusters_center)
            if n > self.Microcluster_threshold:
                #print("label_great", np.array(mic_clusters)[:, self.data_dim])
                # print("label_great_t", np.array(clusters_t)[:, 27])

                ################################################################
                Similarity_all = Similarity_all + Similarity
                Similarity_index_all = Similarity_index_all + Similarity_index
                ################################################################

                if len(clusters_labels) != 0:
                    clusters_outliers_Gclu = np.concatenate((clusters_outliers_Gclu, np.array(clusters_labels)), axis=0)



                if len(clusters_t) != 0:
                    clusters_corrects = np.concatenate((clusters_corrects, np.array(clusters_t)[:, 0:self.data_dim]), axis=0)
                    clusters_y_corrects = np.concatenate((clusters_y_corrects, np.array(clusters_t_y)),
                                                         axis=0)  # 并入正常数据集
            else:
                # 孤立点，小簇的筛选
                # if len(clusters_labels) != 0:
                #     print(np.array(clusters_labels).shape)
                #     print("label_little", np.array(clusters_labels)[:, self.data_dim])

                # if len(clusters_t) != 0:
                # print("label_little_t", np.array(clusters_t)[:, 27])
                microclusters_collection.append(mic_clusters)

                # 并入离群点
                clusters_y_outliers.append(mic_clusters_y)
        clusters_outliers_Gclu = np.delete(clusters_outliers_Gclu, 0, axis=0)
        #print("############107############################################", len(Similarity_all))
        # 得到大簇内相似度小于阈值最小的14个标签
        # if len(Similarity_all) != 0:
        #     S_index = self.get_S_index(Similarity_all, 30)
        #     for i in S_index:
        #         print(Similarity_all[i], Similarity_index_all[i])
        # pd.Series(Similarity_all).sort_values()
        # pd.Series(Similarity_all).sort_values().index[:10]
        # print("-------------------------------------------------------------------------------------------------------")
        # print("shape", clusters_corrects.shape, clusters_y_corrects.shape, len(microclusters_collection),
        #       len(clusters_y_outliers), np.array(microclusters_collection[2]).shape,
        #       np.array(clusters_y_outliers[2]).shape)
        return clusters_corrects, clusters_y_corrects, microclusters_collection, clusters_y_outliers, clusters_outliers_Gclu

    def Microclusters_count(self,Microclusters_out):
        n = 0
        for i in range(len(Microclusters_out)):
            if np.array(Microclusters_out[i]) == -1 or np.array(Microclusters_out[i]) == -2:
                n = n + 1
        return n, len(Microclusters_out)

    def Great_clusters_outliers(self, clusters_outliers_Gclu):

        # 接收大簇里小于相似度大阈值的数据
        # labels_outliers异常点下标
        # print("114", np.array(clusters_outliers_Gclu[2]).shape)
        # print(np.array(clusters_outliers_Gclu[2]))
        # print(len(clusters_outliers_Gclu))
        # print(np.array(clusters_outliers_Gclu)[:, 27])
        labels_outliers = []
        # for i in range(len(clusters_outliers_Gclu)):
        labels_outliers = labels_outliers + (np.array(clusters_outliers_Gclu)[:, self.data_dim]).tolist()
        return labels_outliers

    def get_RMSE(self, y_int, y_out):
        R = np.sqrt(((y_int - y_out) ** 2).mean())
        #print("RMSE_t:", (((y_int - y_out) ** 2)).shape)
        return R

    def Meticulous_screening(self, microclusters_collection, cluster_y_outliers, clusters_corrects, clusters_y_corrects,
                             RMSE_correct, labels_outliers):
        # 完成细筛的工作
        # 区分出微簇和孤立点

        X_outliers = np.ones((1, self.norm_data.shape[1] - 1))
        Y_outliers = np.ones((1, self.norm_data.shape[1] - 1))

        # Correct_microcluster_outliers存放微簇内正确的微簇异常点个数
        Correct_microcluster_outliers = 0
        # Micclustered_data统计微簇数据的总数
        Micclustered_data = 0
        #存放筛选出的孤立点中真正的孤立点异常
        Correct_Orphaned_outliers = []
        # Orphaned_data_collection存放孤立点集合（标签）
        Orphaned_data_collection = []

        # 存放细筛后离群数据的标签
        for i in range(len(microclusters_collection)):
            # print("108", np.array(cluster_outliers[i])[:, 0:27].shape, np.array(cluster_y_outliers[i]).shape)
            #print( np.array(cluster_y_outliers[i]))
            R_o = self.get_RMSE(np.array(microclusters_collection[i])[:, 0:self.data_dim],
                                np.array(cluster_y_outliers[i]))  # 每一小簇离群点的RMSE

            # print("-------------Ro----------------", R_o)
            # if R_o <= 3*RMSE_correct:
            # if R_o <= self.R_threshold:
            #     #print("R", R_o)
            #     # print("已筛出", RMSE_correct)
            #     clusters_corrects = np.concatenate(
            #         (clusters_corrects, np.array(microclusters_collection[i])[:, 0:self.data_dim]),
            #         axis=0)
            #     # print("203", np.array(microclusters_collection[i])[:, self.data_dim])
            #     clusters_y_corrects = np.concatenate((clusters_y_corrects, np.array(cluster_y_outliers[i])), axis=0)

            #else:
                #print("R_o", R_o)
            # X_outliers = np.concatenate((X_outliers, np.array(microclusters_collection[i])[:, 0:self.data_dim]),
            #                             axis=0)
            # Y_outliers = np.concatenate((Y_outliers, np.array(cluster_y_outliers[i])), axis=0)
            #
            # if len(microclusters_collection[i]) > 1:
            #     # 筛选出的异常点小簇
            #     # print("138", type(cluster_outliers), np.array(cluster_outliers[i]).shape)
            #     Correct_microcluster_outliers = Correct_microcluster_outliers + self.Microclusters_count(
            #         np.array(microclusters_collection[i])[:, self.data_dim])[0]
            #
            #     Micclustered_data = Micclustered_data + self.Microclusters_count(
            #         np.array(microclusters_collection[i])[:, self.data_dim])[1]
            # elif len(microclusters_collection[i]) == 1:
            #     # 探测到的所有的孤立点的标签
            #     Orphaned_data_collection.append(np.array(microclusters_collection[i])[:, self.data_dim])
            #     if np.array(microclusters_collection[i])[:, self.data_dim] == -2:
            #         Correct_Orphaned_outliers.append(np.array(microclusters_collection[i])[:, self.data_dim])
            #
            # labels_outliers = labels_outliers + (np.array(microclusters_collection[i])[:, self.data_dim]).tolist()

            R_o = self.get_RMSE(np.array(microclusters_collection[i])[:, 0:self.data_dim],
                                np.array(cluster_y_outliers[i]))  # 每一小簇离群点的RMSE

            #print("-------------Ro----------------", R_o)
            #if R_o <= 3*RMSE_correct:
            if R_o <= self.R_threshold:
                # print("R", R_o)
                # print("已筛出", RMSE_correct)
                clusters_corrects = np.concatenate(
                    (clusters_corrects, np.array(microclusters_collection[i])[:, 0:self.data_dim]),
                    axis=0)
                # print("203", np.array(microclusters_collection[i])[:, self.data_dim])
                clusters_y_corrects = np.concatenate((clusters_y_corrects, np.array(cluster_y_outliers[i])), axis=0)

            else:
                # print("R_o", R_o)
                X_outliers = np.concatenate((X_outliers, np.array(microclusters_collection[i])[:, 0:self.data_dim]),
                                            axis=0)
                Y_outliers = np.concatenate((Y_outliers, np.array(cluster_y_outliers[i])), axis=0)

                if len(microclusters_collection[i]) > 1:
                    # 筛选出的异常点小簇
                    # print("138", type(cluster_outliers), np.array(cluster_outliers[i]).shape)
                    Correct_microcluster_outliers = Correct_microcluster_outliers + self.Microclusters_count(
                        np.array(microclusters_collection[i])[:, self.data_dim])[0]

                    Micclustered_data = Micclustered_data + self.Microclusters_count(
                        np.array(microclusters_collection[i])[:, self.data_dim])[1]
                elif len(microclusters_collection[i]) == 1:
                    # 探测到的所有的孤立点的标签
                    Orphaned_data_collection.append(np.array(microclusters_collection[i])[:, self.data_dim])
                    if np.array(microclusters_collection[i])[:, self.data_dim] == -2:
                        Correct_Orphaned_outliers.append(np.array(microclusters_collection[i])[:, self.data_dim])

                labels_outliers = labels_outliers + (np.array(microclusters_collection[i])[:, self.data_dim]).tolist()
        # print("196", Orphaned_data_collection, Correct_Orphaned_outliers)
        # print("138", Correct_microcluster_outliers, Micclustered_data)
        # print("--------------------------------------labels_outliers--------------------------")
        # print("111", RMSE_correct)
        # print(labels_outliers)
        return clusters_corrects, clusters_y_corrects, labels_outliers, Correct_microcluster_outliers, Micclustered_data, Orphaned_data_collection, Correct_Orphaned_outliers, X_outliers, Y_outliers
