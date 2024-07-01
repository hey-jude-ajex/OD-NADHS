import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from sklearn import decomposition
import tensorflow as tf

def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))  # sort by colume from big to small
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C
    return Cp


def post_proC(C, K, d, alpha):

    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    #spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    #spectral.fit(L)
    #grp = spectral.fit_predict(L)
    ##################################################################################


    return L

def S(C, sigma):


    # L2 normalize
    C = tf.math.l2_normalize(C, axis=1)

    dot_product = tf.matmul(C, C, transpose_b=True)

    # 基于点积矩阵计算高斯核
    similarity = tf.exp(-1 / (2 * (sigma ** 2)) * (1 - dot_product))
    #similarity = tf.exp(-dist_squared / (2 * sigma ** 2))
    return similarity

def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))  # 邻接矩阵

    W = np.sum(C, axis=0)  # 按列求和, 按行求也行
    W = np.diag(1.0/W)  # 是一个1维数组时，形成一个以一维数组为对角线元素的矩阵；是一个二维数组时，输出矩阵的对角线元素
    L = W.dot(C)

    #W = np.sum(C, axis=0)  # 按列求和, 按行求也行
    #W = np.diag(W)  # 是一个1维数组时，形成一个以一维数组为对角线元素的矩阵；是一个二维数组时，输出矩阵的对角线元素
    #L = C - W


    #spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    #spectral.fit(L)
    #grp = spectral.fit_predict(L)
    return L