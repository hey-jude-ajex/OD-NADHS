import numpy
import numpy as np
import pandas as pd
import csv
#全局变量
global x_mat
#data_name = "KDDCUP_normal"
#data_name = "KDDCUP_mscred_data"
#data_name = "change_point_training_3"
#data_name = "kddcup_train"
data_name = "data_7"
data_source_path = "../../data/KDDCUP_t/"+ data_name + ".csv"
#数据标准化
def ZscoreNormalization(x, n):
    meanValue = np.mean(x)
    stdValue = np.std(x)
    print(len(x))
    i = 0
    while i<len(x):
        if stdValue ==0:
            x_mat[i][n] = 0
        else:x_mat[i][n] = (x[i] - meanValue) / stdValue

        #if x_mat[i][n]>0:
        #    print(x_mat[i][n])
        i = i + 1
    print("The ", n , "feature  is normal.")

#-------------------------------------读取文件划分数据集-----------------------------------------
fr = open(data_source_path)
#data_file = open("corrected_normal-result-minmax.csv",'wb+')
lines = fr.readlines()
data = pd.read_csv(data_source_path)

print("28", np.array(data).shape)
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

#标准化处理
ZscoreNormalization(x_mat[:, 0], 0)    #duration
ZscoreNormalization(x_mat[:, 1], 1)
ZscoreNormalization(x_mat[:, 2], 2)
ZscoreNormalization(x_mat[:, 3], 3)
ZscoreNormalization(x_mat[:, 4], 4)    #src_bytes
ZscoreNormalization(x_mat[:, 5], 5)    #dst_bytes
ZscoreNormalization(x_mat[:, 7], 7)    #wrong_fragment
ZscoreNormalization(x_mat[:, 8], 8)    #urgent

ZscoreNormalization(x_mat[:, 9], 9)    #hot
ZscoreNormalization(x_mat[:, 10], 10)  #num_failed_logins
ZscoreNormalization(x_mat[:, 11], 11)
ZscoreNormalization(x_mat[:, 12], 12)  #num_compromised
ZscoreNormalization(x_mat[:, 13], 13)
ZscoreNormalization(x_mat[:, 14], 14)  #su_attempte
ZscoreNormalization(x_mat[:, 15], 15)  #num_root
ZscoreNormalization(x_mat[:, 16], 16)  #num_file_creations
ZscoreNormalization(x_mat[:, 17], 17)  #num_shells
ZscoreNormalization(x_mat[:, 18], 18)  #num_access_files
ZscoreNormalization(x_mat[:, 19], 19)  #num_outbound_cmds
ZscoreNormalization(x_mat[:, 20], 20)
ZscoreNormalization(x_mat[:, 21], 21)
ZscoreNormalization(x_mat[:, 22], 22)  #count
ZscoreNormalization(x_mat[:, 23], 23)  #srv_count
ZscoreNormalization(x_mat[:, 24], 24)  #serror_rate
ZscoreNormalization(x_mat[:, 25], 25)  #srv_serror_rate
ZscoreNormalization(x_mat[:, 26], 26)  #rerror_rate
#ZscoreNormalization(x_mat[:, 27], 27)  #srv_rerror_rate


#chenge_point点处数据的处理
#numpy.savetxt('../../data/data_cut/change_point_data_progressing/'+data_name+'_normal.csv', x_mat, delimiter = ',')
#非change_point数据处理
numpy.savetxt('../../data/KDDCUP_t/data_progressing/'+data_name+'_normal.csv', x_mat, delimiter = ',')