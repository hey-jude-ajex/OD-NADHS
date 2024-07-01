import numpy
import numpy as np
import pandas as pd
import csv
#全局变量

file_name = "a_"
#数据标准化
def ZscoreNormalization(x, n):
    print("10", type(x), x.shape)
    meanValue = np.mean(x)
    print("11", meanValue)
    stdValue = np.std(x)
    print("13", stdValue)
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

source_path = "../../data/stock_t/time-series-todigit/"
save_path = "../../data/stock_t/"
#Change_point_path = "../../data/stock/change_point_train_data/"
#NChange_path = "../../data/stock/"

def run_Normalization(data_name, index):
    global x_mat
    data_path = source_path + data_name + ".csv"
    print(data_path)
    fr = open(data_path)
    # data_file = open("corrected_normal-result-minmax.csv",'wb+')
    lines = fr.readlines()
    data = pd.read_csv(data_path)

    print("28", np.array(data).shape)
    line_nums = len(lines)
    print(line_nums)

    # 创建line_nums行 para_num列的矩阵
    x_mat = np.zeros((line_nums, data.shape[1]))
    n = 0
    # 划分数据集
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = line.split(',')
        x_mat[i, :] = item_mat[0:data.shape[1]]  # 获取42个特征

    fr.close()
    print("45", x_mat.shape, x_mat)

    # --------------------------------获取某列特征并依次标准化并赋值-----------------------------
    print("47", len(x_mat[:, 0]))  # 获取某列数据 494021
    print(len(x_mat[0, :]))  # 获取某行数据 42
    for i in range(data.shape[1]-1):
        n = n+1
        ZscoreNormalization(x_mat[:, i], i)
    print("54", n, x_mat.shape)
    numpy.savetxt(save_path + 'time-series-toNorm/' + data_name + '_normal.csv', x_mat, delimiter=',')


#-------------------------------------读取文件划分数据集-----------------------------------------

# if __name__ == "__main__":
#     for i in range(1):
#         #data_name = "a_CP_" + str(i+4)
#         data_name = "a_" + str(i + 20)
#         index = 1
#         #run_Normalization(data_name, Change_point_path)
#         run_Normalization(data_name, index)
#fr = open("../../data/stock/time-series-todigit/a_" + data_path + ".csv")

if __name__ == "__main__":

    #data_name = "a_CP_" + str(i+4)
    #data_name = "stock_train"
    data_name = "a_6"
    index = 0
    #run_Normalization(data_name, Change_point_path)
    run_Normalization(data_name, index)