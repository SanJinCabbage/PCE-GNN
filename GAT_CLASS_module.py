"""构建GAT模块用于异界邻居信息发掘聚合"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_multi_layer import GraphSecondModuleAttentionLayer

class GAT_CLASS_Model(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_CLASS_Model, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphSecondModuleAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        self.out_att = GraphSecondModuleAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1) # x的形状是(N , 3*out_feature);(N是节点数，out_feature是nhid)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x #这里的输出形状是N * nfeat (相当于还原了)



