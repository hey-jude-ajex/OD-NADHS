import numpy
import numpy as np
import pandas as pd



import pandas as pd
import random
def dbscan(data_set, eps, min_pts, min_outliers):
     #eps:邻域半径
     #min_pts:密度阈值
     examples_nus = np.shape(data_set)[0]  # 样本数量
     unvisited = [i for i in range(examples_nus)]  # 未被访问的点
     visited = []  # 已被访问的点
     # cluster为输出结果，表示对应元素所属类别
     # 默认是一个长度为examples_nus的值全为-1的列表，-1表示噪声点
     cluster = [-1 for i in range(examples_nus)]

     k = - 1  # 用k标记簇号，如果是-1表示是噪声点
     while len(unvisited) > 0:  # 只要还有没有被访问的点就继续循环
         p = random.choice(unvisited)  # 随机选择一个未被访问对象
         unvisited.remove(p)
         visited.append(p)

         nighbor = []  # nighbor为p的eps邻域对象集合，密度直接可达
         for i in range(examples_nus):
             if i != p and np.sqrt(np.sum(np.power(data_set[i, :]-data_set[p, :], 2))) <= eps:  # 计算距离，看是否在邻域内
                 nighbor.append(i)
                 

         if len(nighbor) >= min_pts:  # 如果邻域内对象个数大于min_pts说明是一个核心对象
             k = k+1
             cluster[p] = k  # 表示p它属于k这个簇

             for pi in nighbor:  # 现在要找该邻域内密度可达
                 if pi in unvisited:
                     unvisited.remove(pi)
                     visited.append(pi)

                     # nighbor_pi是pi的eps邻域对象集合
                     nighbor_pi = []
                     for j in range(examples_nus):
                         if np.sqrt(np.sum(np.power(data_set[j]-data_set[pi], 2))) <= eps and j != pi:
                             nighbor_pi.append(j)

                     if len(nighbor_pi) >= min_pts:  # pi是否是核心对象，通过他的密度直接可达产生p的密度可达
                         print("G", pi)
                         for t in nighbor_pi:
                             if t not in nighbor:
                                 nighbor.append(t)

                     if len(nighbor_pi) < min_pts:   #pi不是核心对象且密度小于给定阈值
                         cluster[pi] = -3
                         print("G_index", pi)



                 if cluster[pi] == -1:  # pi不属于任何一个簇，说明第pi个值未改动
                     cluster[pi] = k

         elif len(nighbor) == 1:
             #孤立点
             print("i_index:", p)
             cluster[p] = -2
         else:
             cluster[p] = -1  # 不然就是一个噪声点
     return cluster
#def get_result(data, label):

data_path = "./SWaT_1.csv"
data = pd.read_csv(data_path, header=None)
data = np.array(data)
label = dbscan(data, 0.1, 40, 20)
numpy.savetxt("../../data/SWaT/SWaT_label_pre_1.csv", label, delimiter=',')
print(label)
label_t = np.array(label, dtype=bool)
numpy.savetxt("../../data/SWaT/SWaT_label_pre_1_t.csv", label_t, delimiter=',')

y = pd.read_csv("./SWaT_label_1.csv", header=None)

y_pre = pd.read_csv("./SWaT_label_pre_1_t.csv", header=None)
print(y.shape, y_pre.shape)
y_pre = np.array(y_pre, dtype=bool)
y = np.array(y,dtype=bool)
#numpy.savetxt("../../data/SWaT/SWaT_label_pre_1_t.csv", y, delimiter=',')

print(y, y_pre)
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


R = recall_score(y, y_pre)
P = precision_score(y, y_pre)
F1 = f1_score(y, y_pre)
print(R, P, F1)





