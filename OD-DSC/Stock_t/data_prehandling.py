import numpy
import numpy as np
import pandas as pd
import csv
#全局变量
global x_mat
#读取txt里的文件

file_name = "a_1"

def data_load(data_path):
    '''
    data = []
    file = open(data_path, 'r')
    file_data = file.readlines()
    for row in file_data:
        tmp_list = row.split(',')
        tmp_list[-1] = tmp_list[-1]
        data.append(tmp_list)
    '''
    data = np.loadtxt(data_path, dtype=np.float, delimiter=',')
    return  data

if __name__ == "__main__":
    for i in range(1):
        data_label = str(20)
        data_path = "../../data/stock/time-series/a_" + data_label + ".txt"

        data = np.array(data_load(data_path))
        #print(data, type(data), np.array(data).shape, np.array(data))
        #添加标签列
        print("32", np.array(data).shape)
        a = np.ones(data.shape[0])
        print("34", a.shape)
        data = np.column_stack((data, a))
        print("34", data.shape)
        numpy.savetxt('../../data/stock/time-series-todigit/a_' + data_label + '.csv', data, delimiter=',')