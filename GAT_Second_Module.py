"""构建GAT模块用于异界邻居信息发掘聚合"""
import torch
import torch.nn as nn
from GAT_CLASS_module import GAT_CLASS_Model
class LDSN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, combine_arr):
        """
        :param nfeat:  节点特征维度
        :param node_type_num: 节点的种类个数 （每个种类队医一层attention_layer）
        :param nhid: 隐藏层个数
        :param dropout: 损失率
        :param alpha: 什么玩意儿带查询
        :param nheads: 几个头，每层有几个注意力头关注度吧
        """
        """Dense version of GAT."""
        super(Classify_Second_Module, self).__init__()
        self.dropout = dropout
        # 开始对数据进行分类，并把不同种类的数据输入到不同的分类器中。

        self.combine_arr = combine_arr # combier_arr 是一个包含多种类型节点的数组【[],[],[],[]】每个数组代表着不同的种类，

        self.attentions = [GAT_CLASS_Model(nfeat, nhid, nclass, dropout, alpha, nheads) for _ in range(len(self.combine_arr))]

        for i, attention in enumerate(self.attentions):
            self.add_module('different_class_model_attention_{}'.format(i), attention)

    def forward(self):

        x = [self.attentions[index](self.combine_arr[index]["feature"], self.combine_arr[index]["adj"]) for index in range(len(self.attentions))]
        # x = 7 ge 2078 * 7
        # 接下来的工作思路
        # 1. x其实是已经包含绝大部分节点的这么一种数组, 它的index就代表的是不同的节点
        # 2. 存表的时候其实有一个id列表，不同的分类数组对应不同的id列表，可以依据这个对节点进行重构 有就有，没有就算了
        # final = x[0]
        # for index in range(len(x)-1):
        #     final = torch.max(final, x[index + 1])
        return sum(x)

