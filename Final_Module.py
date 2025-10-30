"""构建GAT模块用于异界邻居信息发掘聚合"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_k_mean import K_Means_Classify_Fn as KM
from k_means import K_Means_Classify_Fn
from GAT_module import LOCAL
from GAT_Second_Module import LDSN


class Model_Framework(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(Model_Framework, self).__init__()
        self.dropout = dropout

        self.combine_arr = K_Means_Classify_Fn(nclass)

        self.LOCAL = LOCAL(nfeat, 5, nclass, dropout, alpha, nheads)

        self.LDSN = LDSN(nfeat, 5, nclass, dropout, alpha, nheads, self.combine_arr)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.LOCAL(x, adj)
        x2 = self.LDSN()
        Final = F.leaky_relu(torch.max(x1, x2))

        return F.log_softmax(Final, dim=1)


