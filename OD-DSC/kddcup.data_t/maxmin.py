#coding:utf-8
import numpy
import numpy as np
import pandas as pd
import csv
#data_name = 'KDDCUP_normal'
data_name = 'data_29'
#data_name = "change_point_training_3"
#data_source_path = "../../data/data_cut/change_point_data_progressing/"+ data_name + "_normal.csv"
#非change_point数据点
data_source_path = "../../data/KDDCUP_t/data_progressing/"+ data_name + "_normal.csv"
#全局变量
global x_mat

#数据归一化
def MinmaxNormalization(x, n):
    minValue = np.min(x)
    maxValue = np.max(x)
    print(minValue, maxValue)
    print(len(x))
    i = 0
    while i<len(x):
        if maxValue - minValue ==0:
            x_mat[i][n] =0
        else:x_mat[i][n] = (x[i] - minValue) / (maxValue - minValue)
        #if x_mat[i][n]>0:
        #    print(x_mat[i][n])
        i = i + 1
    print("The ", n , "feature  is normal.")

#-------------------------------------读取文件划分数据集-----------------------------------------
fr = open(data_source_path)
data = pd.read_csv(data_source_path)
print("29", np.array(data).shape)
#data_file = open("corrected_normal-result-minmax.csv",'wb+')
lines = fr.readlines()
line_nums = len(lines)
print(line_nums)

#创建line_nums行 para_num列的矩阵
x_mat = np.zeros((line_nums, 28))

#划分数据集
for i in range(line_nums):
    line = lines[i].strip()
    item_mat = line.split(',')
    x_mat[i, :] = item_mat[0:28]    #获取42个特征
fr.close()
print(x_mat.shape)

#--------------------------------获取某列特征并依次标准化并赋值-----------------------------
print(len(x_mat[:, 0])) #获取某列数据 494021
print(len(x_mat[0, :])) #获取某行数据 42

#归一化处理
MinmaxNormalization(x_mat[:, 0], 0)    #duration
MinmaxNormalization(x_mat[:, 1], 1)
MinmaxNormalization(x_mat[:, 2], 2)
MinmaxNormalization(x_mat[:, 3], 3)
MinmaxNormalization(x_mat[:, 4], 4)    #src_bytes
MinmaxNormalization(x_mat[:, 5], 5)    #dst_bytes
MinmaxNormalization(x_mat[:, 6], 7)
MinmaxNormalization(x_mat[:, 7], 7)    #wrong_fragment
MinmaxNormalization(x_mat[:, 8], 8)    #urgent

MinmaxNormalization(x_mat[:, 9], 9)    #hot
MinmaxNormalization(x_mat[:, 10], 10)  #num_failed_logins
MinmaxNormalization(x_mat[:, 11], 11)
MinmaxNormalization(x_mat[:, 12], 12)  #num_compromised
MinmaxNormalization(x_mat[:, 13], 13)
MinmaxNormalization(x_mat[:, 14], 14)  #su_attempte
MinmaxNormalization(x_mat[:, 15], 15)  #num_root
MinmaxNormalization(x_mat[:, 16], 16)  #num_file_creations
MinmaxNormalization(x_mat[:, 17], 17)  #num_shells
MinmaxNormalization(x_mat[:, 18], 18)  #num_access_files
MinmaxNormalization(x_mat[:, 19], 19)  #num_outbound_cmds
MinmaxNormalization(x_mat[:, 20], 20)
MinmaxNormalization(x_mat[:, 21], 21)
MinmaxNormalization(x_mat[:, 22], 22)  #count
MinmaxNormalization(x_mat[:, 23], 23)  #srv_count
MinmaxNormalization(x_mat[:, 24], 24)  #serror_rate
MinmaxNormalization(x_mat[:, 25], 25)  #srv_serror_rate
MinmaxNormalization(x_mat[:, 26], 26)  #rerror_rate
#MinmaxNormalization(x_mat[:, 27], 27)



#numpy.savetxt('../../data/data_cut/change_point_data_progressing/'+data_name + '_normal_minmax.csv', x_mat, delimiter = ',')
#非change_point点
numpy.savetxt('../../data/KDDCUP_t/data_progressing/'+data_name+'_normal_minmax.csv', x_mat, delimiter = ',')
#numpy.savetxt('../../data/KDDCUP/kddcup_Normal.csv', x_mat, delimiter = ',')

'''
#文件写入操作
csv_writer = csv.writer(data_file)
i = 0
while i<len(x_mat[:, 0]):
    csv_writer.writerow(x_mat[i, :])
    i = i + 1
data_file.close()
'''