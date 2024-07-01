import numpy as np
import pandas as pd
import json


parameters = {
        #网络层结构
        'n_hidden_enc_1':[7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7,   7,7,7,7,7,  7,7],
        'n_hidden_enc_2':[7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7,   7,7,7,7,7,  7,7],
        'n_hidden_dec_2':[7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7,   7,7,7,7,7,  7,7],
        'n_hidden_dec_1':[7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7,   7,7,7,7,7,  7,7],
        #训练DSC损失的参数
        'alpha_dsc_a1':[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,   1,1,1,1,1,  1,1],
        'alpha_dsc_a2':[10,10,10,10,10, 10,10,10,10,10, 10,10,10,10,10, 10,10,10,10,10,    10,10,10,10,10,  10,10],
        #确定相似度矩阵的参数：
        'ro':   [0.35,0.35,0.35,0.35,0.35,      0.35,0.35,0.35,0.35,0.35,      0.35,0.35,0.35,0.4,0.4,        0.4,0.478,0.478,0.51,0.51,   0.4,0.4,0.4,0.4,0.4,  0.4,0.4],
        'alpha':[0.85,0.85,0.85,0.85,0.85, 0.85,0.85,0.85,0.85,0.85,  0.85,0.85,0.85,0.6,0.6, 0.6,0.85,0.85,0.73,0.73,   0.85,0.85,0.85,0.85,0.85,  0.85,0.85],
        'd':[7,7,7,7,7,                     7,7,7,7,7, 7,7,7,7,7, 7,7,7,7,7,    7,7,7,7,7,  7,7],
        #聚簇的参数
        'Similarity_threshold':   [0.65,0.65,0.65,0.65,0.65,           0.65,0.65,0.65,0.65,0.65,       0.65,0.65,0.65,0.65,0.65,  0.65,0.65,0.65,0.65,0.65,   0.65,0.65,0.65,0.65,0.65,   0.65,0.65],  # 相似度阈值
        'Large_cluster_threshold':[0.75,0.75,0.75,0.75,0.75,      0.75,0.75,0.75,0.75,0.75, 0.75,0.75,0.75,0.75,0.75,   0.75,0.75,0.75,0.75,0.75,    0.75,0.75,0.75,0.75,0.75,  0.75,0.75],    #大簇阈值
        'Microcluster_threshold':[9,9,9,9,9,                 9,9,9,9,9,                9,9,9,9,9,                9,9,9,9,9,   9,9,9,9,9,  9,9],
        #R_threshold:细筛部分用到的RMSE阈值6
        'R_threshold':[0.165,0.165,0.165,0.165,0.165, 0.165,0.165,0.165,0.165,0.165, 0.165,0.165,0.165,0.12,0.12,  0.12,0.16,0.16,0.2,0.2,    15,0.15,0.15,0.15,0.15,   0.15,0.15],
        #Gco_Q: Great_clusters_outliers_quantity
        'Gco_Q':[26,32,19,27,24, 29,8,4,15,19, 23,13,15,2,2,         2,0,0,0,0,   0,0,0,0,0,  0,0],
        'C_point_index':[1,1,1,1,1, 1,1,1,1,1, 1,1,1,14,14, 14,17,17,17,17,   21,22,23,24,25,  26,27]
        }
pd.DataFrame(parameters)


df = pd.DataFrame(parameters)
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



#'C_point_index':[0,0,0,0,0, 1,1,1,1,1, 2,2,2,3,3, 3,4,4,4,4]