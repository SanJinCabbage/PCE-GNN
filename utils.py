import numpy
import numpy as np
import scipy.sparse as sp
import torch
import math
import torch.nn.functional as F

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    cn_labels = idx_features_labels[:, -1]
    # ids = idx_features_labels[:,0]
    # org_feature = idx_features_labels[:, 1:-1]

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(1354)
    idx_val = range(1355, 2221)
    idx_test = range(2222, 2707)


    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_origin(path="./data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    cn_labels = idx_features_labels[:, -1]
    ids = idx_features_labels[:, 0]
    org_feature = idx_features_labels[:, 1:-1]
    return cn_labels, ids, org_feature

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def print_red(text):
    return "\033[31m" + text + "\033[0m"


def FindCenter(Points, Mean_Center):
    # 初始化最小距离和最近的点
    min_distance = float('inf')
    nearest_point = None

    # 遍历所有点
    for point in Points:

        # 计算距离 3維
        x, y, z= point
        distance = math.sqrt((x - Mean_Center[0]) ** 2 + (y - Mean_Center[1]) ** 2 + (z - Mean_Center[1]) ** 2 )

       # 更新最小距离和最近的点
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
    return nearest_point

# 查找节点再adj中的位置
def Find_LOCATION(POINT,IDS):
    index = 0
    key_list = IDS[:, 0]
    for j in range(len(IDS)):
        if numpy.any(numpy.isclose(key_list[j],POINT)):
            index = j
    return index

def Fiber_LOCATION(POINTS,IDS,key):

    IndexList = []
    POINTS = np.array(POINTS)
    pointList = POINTS[:,0]
    for i in range(len(pointList)):
        IndexList.append([Find_LOCATION(pointList[i],IDS),key[Find_LOCATION(POINTS[i], IDS)]])

    return IndexList


def Fiber_Edge(arr_1, arr_2):
    new = numpy.array(arr_1)

    for i in range(len(arr_1)):
        if arr_1[i] == 0 and numpy.isin(i, arr_2):
            new[i] = 1
        # elif arr_1[i] != 0 and not numpy.isin(i, arr_2):
        #     new[i] = 0
    return new


def accuracy(output, labels):
    """
    计算精确度
    :param output: 形如：【【a,b,c,d】,【a,b,c,d】【a,b,c,d】】，a,b,c,d分别对应某一类的概率
    :param labels: [0,1,2]
    :return: 准确度
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def array_to_adjacency_matrix(array, index=0):
    n = len(array)
    # 初始化一个n x n的零矩阵
    adj_matrix = np.zeros((n, n), dtype=int)
    arrays = np.array(array, dtype=int)
    # 遍历数组，设置邻接矩阵
    for i in range(n):
        if arrays[i] != 0:
            # 假设数组中的1表示节点i与节点i+k有连接（这里k=1表示简单的线性连接）
            # 对于无向图，我们假设每个1表示一个边，并且边连接的是相邻的索引（循环连接）
            # 例如，对于数组 [0, 1, 0, 1]，我们可以假设它表示一个环：0-1-2-3-0
            # 这里为了简单起见，我们假设每个1只与相邻的1（或首尾相连）有边
            # 如果需要更复杂的连接规则，请修改此部分

            # 简单的线性连接（首尾相连）
            # j = index % n  # 下一个索引，循环到开头
            adj_matrix[index, i] = 1
            adj_matrix[i, index] = 1  # 因为是无向图，所以对称

    # 注意：这个简单的线性连接假设可能不适用于所有情况，
    # 根据你的具体需求，你可能需要调整连接逻辑。

    return np.array(adj_matrix)

def FiberFeatures(array,feature):
    arrays = np.array(array, dtype=int)
    result = []
    for i in range(len(arrays)):
        if arrays[i] == 0:
            result.append(np.zeros(shape=(feature.shape[1],),dtype=np.float32))
        else:
            result.append(feature[i].astype(np.float32))
    return np.array(result)
# 示例数组
# array = np.array([0, 1, 1, 1])
# adj_matrix = array_to_adjacency_matrix(array,1)
# print(adj_matrix)

def K_Means_Classify_Fn():
    class_1 = {"X":   torch.tensor([
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0]],
        dtype=torch.float32,requires_grad=True),
               "adj": torch.tensor([
                   [1, 1],
                   [1, 1]])
               }
    class_2 = {"X":   torch.tensor([
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0]],
        dtype=torch.float32,requires_grad=True),
               "adj": torch.tensor([
                   [1, 1],
                   [1, 1]])
               }
    class_3 = {"X":   torch.tensor([
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0]],
        dtype=torch.float32,requires_grad=True),
               "adj": torch.tensor([
                   [1, 1],
                   [1, 1]])
               }
    temp = [class_1,class_2,class_3]
    return temp

def calculate_precision(softmax_output, labels, acc_class, threshold=0.5):
    softmax_output = softmax_output.cpu().detach().numpy()
    """
    计算精确率（Precision）
    参数:
    softmax_output (numpy.ndarray): 经过softmax函数输出的n*7张量矩阵，表示每个样本属于每个类别的概率。
    labels (numpy.ndarray): 真实标签数组，长度为n，每个元素是0到6之间的整数。
    threshold (float): 用于将softmax输出转换为二分类决策（是否预测为正类）的阈值，默认为0.5。

    返回:
    float: 精确率（Precision）
    """
    # 初始化召回率数组，长度为类别数
    precision = np.zeros(acc_class)

    # 将softmax输出转换为预测的类别标签
    predicted_labels = np.argmax(softmax_output, axis=1)

    for i in range(acc_class):
        # 创建一个布尔数组来表示哪些样本被预测为正类
        predicted_as_positive = (predicted_labels == i)  # 假设0是负类，1-6是正类

        # 创建一个布尔数组来表示哪些样本实际上是正类
        actual_as_positive = (labels == i).cpu().detach().numpy()  # 同样假设

        # 计算真正例（TP）
        TP = np.sum((predicted_as_positive & actual_as_positive).astype(int))

        # 计算假正例（FP）
        FP = np.sum((predicted_as_positive & ~actual_as_positive).astype(int))
        # 计算精确率
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0  # 避免除以零


    return torch.tensor(precision)


def calculate_recall(softmax_output, labels, acc_class):
    """
    计算多分类任务的召回率（Recall）

    参数:
    softmax_output (numpy.ndarray): 经过softmax函数输出的n*7张量矩阵，n为样本数，7为类别数。
    labels (numpy.ndarray): 真实标签数组，长度为n。

    返回:
    numpy.ndarray: 每个类别的召回率（Recall），长度为7。
    """
    softmax_output = softmax_output.cpu().detach().numpy()
    # 将softmax输出转换为预测的类别标签
    predicted_labels = np.argmax(softmax_output, axis=1)

    # 初始化召回率数组，长度为类别数
    recall = np.zeros(acc_class)

    # 遍历每个类别
    for i in range(acc_class):
        # 找到所有真实标签为该类别的样本索引
        true_positives_indices = np.where(labels.cpu().detach().numpy() == i)[0]

        # 在这些样本中，找到预测标签也为该类别的样本数量（真正例TP）
        true_positives = np.sum(predicted_labels[true_positives_indices] == i)

        # 计算该类别的召回率
        # 注意：如果该类别的真实样本数为0，则召回率设为0（避免除以零）
        if len(true_positives_indices) > 0:
            recall[i] = true_positives / len(true_positives_indices)
        else:
            recall[i] = 0.0

    return torch.tensor(recall)

def calculate_micro_f1(softmax_output, labels):

    # 获取softmax输出的形状
    softmax_output = softmax_output.cpu().detach().numpy()
    n, num_classes = softmax_output.shape

    # 获取每个样本的预测标签（概率最高的类别）
    predicted_labels = np.argmax(softmax_output, axis=1)

    # 初始化TP, FP, FN计数器
    TP = np.zeros(num_classes, dtype=int)
    FP = np.zeros(num_classes, dtype=int)
    FN = np.zeros(num_classes, dtype=int)

    # 遍历所有样本和类别，计算TP, FP, FN
    for i in range(n):
        true_label = labels[i]
        pred_label = predicted_labels[i]

        if true_label == pred_label:
            TP[true_label] += 1
        else:
            FP[pred_label] += 1
            FN[true_label] += 1

    # 计算Micro-Precision和Micro-Recall
    micro_precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
    micro_recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))

    # 计算Micro-F1
    if micro_precision + micro_recall == 0:
        # 避免除以零的情况
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1


def calculate_macro_f1(softmax_output, labels):
    softmax_output = softmax_output.cpu().detach().numpy()
    # 获取softmax输出的形状
    n, num_classes = softmax_output.shape

    # 获取每个样本的预测标签（概率最高的类别）
    predicted_labels = np.argmax(softmax_output, axis=1)

    # 初始化TP, FP, FN计数器数组，长度为num_classes
    TP = np.zeros(num_classes, dtype=int)
    FP = np.zeros(num_classes, dtype=int)
    FN = np.zeros(num_classes, dtype=int)

    # 遍历所有样本和类别，计算TP, FP, FN
    for i in range(n):
        true_label = labels[i]
        pred_label = predicted_labels[i]

        if true_label == pred_label:
            TP[true_label] += 1
        else:
            FP[pred_label] += 1
            FN[true_label] += 1

    # 计算每个类别的Precision和Recall
    precision = TP / (TP + FP + 1e-10)  # 加1e-10避免除以零
    recall = TP / (TP + FN + 1e-10)  # 加1e-10避免除以零

    # 计算每个类别的F1分数，并取平均值得到Macro-F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # 加1e-10避免除以零
    macro_f1 = np.mean(f1_scores)

    return macro_f1