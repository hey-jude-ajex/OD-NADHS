
import numpy as np
import pandas as pd
import json


parameters = {
        #网络层结构
        'n_hidden_enc_1':[27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27,  27,27,27,27,27,  27,27],
        'n_hidden_enc_2':[20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20,  20,20,20,20,20,  20,20],
        'n_hidden_dec_2':[20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20, 20,20,20,20,20,  20,20,20,20,20,  20,20],
        'n_hidden_dec_1':[27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27,  27,27,27,27,27,  27,27],
        #训练DSC损失的参数
        'alpha_dsc_a1':[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,  1,1,1,1,1,  1,1],
        'alpha_dsc_a2':[2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2,  2,2,2,2,2,  1,1],
        #确定相似度矩阵的参数：
        'ro':   [0.35,0.35,0.35,0.35,0.35,  0.35,0.35,0.35,0.35,0.35,    0.45,0.45,0.45,0.45,0.45,   0.365,0.365,0.365,0.365,0.365, 0.1,0.1,0.1,0.1,0.1, 0.27,0.27,0.27,0.27,0.27,   0.77,0.77,0.77,0.77,0.77,   0.77,0.77],
        'alpha':[0.7,0.7,0.7,0.7,0.7,   0.37, 0.37, 0.37, 0.37, 0.37,     0.6,0.6,0.6,0.6,0.6,   0.9,0.7,0.7,0.7,0.7,    0.9,0.6,0.6,0.6,0.6, 0.4,0.4,0.4,0.4,0.4,   0.7,0.7,0.7,0.7,0.7,  0.7,0.7],
        'd':[27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 27,27,27,27,27, 25,27,27,27,27,  27,27,27,27,27,   27,27],
        #聚簇的参数

        'Similarity_threshold':   [0.66,0.66,0.66,0.66,0.66, 0.66,0.66,0.66,0.66,0.66, 0.66,0.66,0.66,0.66,0.66, 0.66,0.66,0.66,0.66,0.66, 0.66,0.66,0.66,0.66,0.66, 0.66,0.66,0.66,0.66,0.66,   0.66,0.66,0.66,0.66,0.66,   0.66,0.66],  # 相似度阈值
        'Large_cluster_threshold':[0.7,0.7,0.7,0.7,0.7,      0.7,0.7,0.7,0.7,0.7,      0.7,0.7,0.7,0.7,0.7,      0.7,0.7,0.7,0.7,0.7,      0.7,0.7,0.7,0.7,0.7,      0.7,0.7,0.7,0.7,0.7,  0.7,0.7,0.7,0.7,0.7,   0.66,0.66],    #大簇阈值
        'Microcluster_threshold':[8,8,8,8,8,                 8,8,8,8,8,                8,8,8,8,8,                8,8,8,8,8,           8,8,8,8,8 ,                    8,8,8,8,8,   8,8,8,8,8,   8,8],
        #R_threshold:细筛部分用到的RMSE阈值
        'R_threshold':[0.365,0.365,0.365,0.365,0.365, 0.37,0.37,0.37,0.37,0.37, 0.32,0.32,0.32,0.32,0.32, 0.28,0.28,0.28,0.28,0.28, 0.28,0.28,0.28,0.28,0.28, 5.35,5.35,5.35,5.35,5.35,   0.365,0.365,0.365,0.365,0.365,   0.365,0.365],
        #Gco_Q: Great_clusters_outliers_quantity
        'Gco_Q':[0,0,0,0,20, 20,20,20,13,20, 0,0,0,13,0, 0,5,0,0,7, 0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0,   0,0],
        'C_point_index':[1,1,1,1,1, 6,6,6,6,6, 11,11,13,14,14, 16,16,16,16,16, 21,21,21,21,21, 26,26,26,26,26,  31,32,33,34,35,  36,37]
        }

pd.DataFrame(parameters)
#print(parameters['Large_cluster_threshold'][5])

# 转换为DataFrame
df = pd.DataFrame(parameters)

# 保存为json文件
df.to_json('./parameters_dict.json', orient='records', lines=True)

df.head()

# 读取 JSON 文件
df = pd.read_json('./parameters_dict.json', orient='records', lines=True)

# 将 DataFrame 转换为字典列表
data = df.to_dict(orient='records')

# 格式化并保存为易读的 JSON 文件
with open('./parameters_formatted.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# 打印格式化后的 JSON
print(json.dumps(data, ensure_ascii=False, indent=4))


