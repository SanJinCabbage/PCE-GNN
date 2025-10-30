# 导入必要的库
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import utils
import os
os.environ["OMP_NUM_THREADS"] = "2"
# 数据降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from test_load_zb import load_zb_data,load_zb_data_cn
# 使用make_blobs生成聚类数据
# X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# X为one-hot编码后的数据矩阵--方式一
# pca = PCA(n_components=0.80)  # 保留95%的方差
# X_reduced = pca.fit_transform(features_x)

# X为one-hot编码后的数据矩阵--方式三
# tsne = TSNE(n_components=2)  # 降维到6个维度
# X_reduced_third = tsne.fit_transform(X,labels_y)

def DrawGraph(base, noteable, RealNode, color, size, ShowRealNode = False):
    # 绘制原始数据和聚类结果
    plt.scatter(base["x"][:, 0], base["y"][:, 1], c=base["label"], cmap='viridis', edgecolor='k')
    plt.scatter(noteable["x"][:, 0], noteable["y"][:, 1], c=noteable["label"], s=noteable["size"], label='Centroids')
    if RealNode:
       plt.scatter(RealNode["x"][:, 0], RealNode["y"][:, 1], c=RealNode["label"], s=RealNode["size"], label='Centroids')

    # plt.scatter(X_reduced_second[:, 0], X_reduced_second[:, 1], c=label_s, cmap='viridis', edgecolor='k')
    # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, label='Centroids')
    # plt.scatter(center_list[:, 0], center_list[:, 1], c='yellow', s=60, label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

#获取不同类型的数据集合
def ClusterLabel(array, num, ids, features_x, cn_labels):
    """将相同标签的聚类在一起 下标"""
    result = []
    result_full = []
    for i in range(num):
       result.append([])
       result_full.append([])

    for i in range(len(array)):
        result[array[i]].append(i)
        team = []
        team.append(ids[i])
        team.append(features_x[i].tolist())
        team.append(cn_labels[i])
        team = list(itertools.chain.from_iterable(
            [item if isinstance(item, list) else [item] for item in team]
        ))
        result_full[array[i]].append(team)

    return result, result_full


def WriteIntoFile(file_name, data):
    # 将分类好的文件写入到outfles中
    filename = file_name
    # 打开文件以写入模式（如果文件不存在，它将被创建）
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入数组内容到 CSV 文件
        # 如果你想写入表头（列名），可以在这里添加一行 writer.writerow(['Header1', 'Header2', 'Header3'])
        for row in range(len(data)):
            for item in range(len(data[row])):
                writer.writerow(data[row][item])

    print(f"数据已保存到 {filename}")


def Find_Info(arr_1, arr_id, arr_feature, arr_label):
    """
    利用arr_1给出的数据来,从arr_2中找到对应下标的数据，
    :param arr_1: 下标数组--分类好的节点集合
    :param arr_id: 原始信息id集合 --ids
    :param arr_feature:  原始信息特征集合 --org_feature
    :param arr_label:  原始信标签集合 --cn_labels
    :return: 【id,feature,label】
    """
    tep_feature = arr_feature.tolist()
    temp = []
    for i in range(len(arr_1)):
        index = arr_1[i]
        arr_feature_list = arr_feature[index].tolist()
        temp.append([arr_id[index]]+arr_feature_list+[arr_label[index]])

    return temp


def set_cluster_to_file(arr, ids, org_feature, cn_labels):
    final_arr = []
    for i in range(len(arr)):
        final_arr.append(Find_Info(arr[i], ids, org_feature, cn_labels))

    return final_arr


"""K-Means算法分类"""
def K_Means_Classify_Fn(K):
    # 数据准备： 从cora加载需要分类的数据
    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data(path="./data/chamelon/", dataset="chamelon")
    cn_labels, ids, org_feature = utils.load_data_origin(path="./data/chamelon/", dataset="chamelon")

    # MY SelfData
    # cn_labels, ids, org_feature = load_zb_data_cn()
    # adj, features, labels, idx_train, idx_val, idx_test = load_zb_data()

    # 数据处理: 将tensor数据处理成numpy格式方便后面对其进行操作
    features_x = features.numpy()
    labels_y = labels.numpy()

    # 选取对One-hot编码的处理方式： X为one-hot编码后的数据矩阵--方式二
    lda = LinearDiscriminantAnalysis(n_components=3)  # 降维到2个维度
    X_reduced_second = lda.fit_transform(features_x, labels_y)

    # 使用KMeans算法进行聚类
    kmeans = KMeans(n_clusters=K, n_init='auto')
    kmeans.fit(X_reduced_second)

    # 获取聚类结果和聚类中心
    label_s = kmeans.labels_
    centers = kmeans.cluster_centers_
    center_list = [] # 用于存放距离中心节点距离最近的真是节点，中心节点其实并不在真是数据列表中

    # 获取距离聚类中心最近的节点，（这个K-means算法返回的中心点不一定是在节点里面，得想办法把他放到里面）
    for i in range(K):
        center_list.append(utils.FindCenter(X_reduced_second, centers[i]))
    center_list = np.array(center_list)

    # 分类标签
    clusters, clusters_ids = ClusterLabel(label_s, K, ids, features_x, cn_labels)


    #  1. 找到中心节点k 属于 N  val[i]：【节点在cora.content中的序号，下标减去1（因为内阁文件是从1开始的，这个数组是从0开始）,这个节点的id】
    # 例如【1338，562067】 1338是节点下标，562067是id编号
    val = utils.Fiber_LOCATION(center_list, X_reduced_second, ids)

    # utils.Fiber_Array(val[0][0],[],clusters[0])
    #  2. 以k为中心， 对原有的矩阵进行修补, 生成|N|个子adj

    # element = set_cluster_to_file(clusters, ids, features_x, cn_labels)

    # for item in range(len(element)):
    #     file_name = "./template_data/cluster{}.csv".format(item)
    #     with open(file_name, mode='w', newline='') as file:
    #         for j in element[item]:
    #             writer = csv.writer(file)
    #             writer.writerow(j)
    #     print(f"数据已保存到 {file_name}")

    # 将分类好的数据在整理一下整理成这种形状
    # [{"feature": [xxx], "adj": [xxx]},...,{"feature": [xxx], "adj": [xxx]}]

    new_arr = []  # 存放分类好的节点图
    FinalResult = [] # 用于存放最终结果， [{"feature": [], "adj", []}]
    for i in range(len(clusters)):
        new_arr.append(utils.Fiber_Edge(adj[val[i][0]], clusters[i]))
        # adj

        feature_me = utils.FiberFeatures(new_arr[i], org_feature)
        FinalResult.append({
            "adj": torch.from_numpy(utils.array_to_adjacency_matrix(new_arr[i], val[i][0])).cuda(),
            "feature": torch.from_numpy(feature_me).cuda()
        })

    return FinalResult
