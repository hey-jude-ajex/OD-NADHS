import numpy
import numpy as np
import pandas as pd

import numpy as np


global x_mat
data_source_path = "../../data/data_cut/kddcup/data_t.csv"

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
#data_file = open("corrected_normal-result-minmax.csv",'wb+')
lines = fr.readlines()
data = pd.read_csv(data_source_path, header=None)
dim = data.shape[1]

data_t = np.transpose(data)
print("49", data.shape)
print("28", dim)
line_nums = len(lines)


#创建line_nums行 para_num列的矩阵
x_mat = np.zeros((line_nums, data.shape[1]))

#划分数据集
for i in range(line_nums):
    line = lines[i].strip()
    item_mat = line.split(',')
    x_mat[i, :] = item_mat[0:data.shape[1]]    #获取42个特征
fr.close()
print(x_mat.shape)

#--------------------------------获取某列特征并依次标准化并赋值-----------------------------
print(len(x_mat[:, 0])) #获取某列数据 494021
print(len(x_mat[0, :])) #获取某行数据 42

#标准化处理


for i in range(data.shape[1]-1):

    ZscoreNormalization(x_mat[:, i], i)

for i in range(data.shape[1]-1):
    MinmaxNormalization(x_mat[:, i], i)

print("80", data_t.shape)
numpy.savetxt('../../data/data_cut/kddcup/kddcup_data_mscred.csv', data_t, delimiter = ',')
numpy.savetxt('../../data/data_cut/kddcup/data_kdd_normal_minmax.csv', x_mat, delimiter = ',')


