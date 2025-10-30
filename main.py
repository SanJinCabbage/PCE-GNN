import copy
import time
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from Final_Module import Model_Framework
from utils import load_data, accuracy, calculate_precision, calculate_recall,calculate_macro_f1 ,calculate_micro_f1


#  配置数据
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=3, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
args = parser.parse_args()

# 加载数据集合
adj, features, labels, idx_train, idx_val, idx_test = load_data(path="./data/chamelon/", dataset="chamelon")
# Myself data
# adj, features, labels, idx_train, idx_val, idx_test = load_zb_data()
# df = pd.read_csv('./data/squirrel/value.train_masks.csv')
# train_mask = df['train_mask'].values
# train_mask = df['train_mask'].astype(bool).values
# cn_labels, ids, org_feature = load_data_origin(path="./data/CiteSeer/", dataset="CiteSeer")
# train_mask = np.genfromtxt('./data/squirrel/value.train_masks.csv')
# test_mask = np.genfromtxt('./data/squirrel/test_mask.csv')
# val_mask = np.genfromtxt('./data/squirrel/val_mask.csv')


# cora
NCLASS= 7
NFEAT = 1433


# GAT模型实例化
gat_model = Model_Framework(NFEAT, args.hidden, NCLASS, args.dropout, args.alpha, args.nb_heads)

# 引入optimizer工具
optimizer = optim.Adam(gat_model.parameters(), lr=0.008, weight_decay=5e-4)
EPOCH = 1200
BEST_LOSS = float('inf')  # 初始化最佳损失为无穷大
# BEST_MODEL_WTS = torch.load("best_model.pth",weights_only=False)  # 初始化最佳模型权重为None
BEST_MODEL_WTS = None # 初始化最佳模型权重为
USE_CUDA = True
if torch.cuda.is_available() and USE_CUDA:
    gat_model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_test.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_train.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

#  训练函数，最终达到的效果是训练K次返回k次中结果最好权重参数文件
def train(epoch):
    t = time.time()
    gat_model.train()
    optimizer.zero_grad() # 首先清空梯度
    Y = gat_model(features, adj)
    loss_fn = F.nll_loss(Y[idx_train], labels[idx_train]) # 定义损失函数
    acc_train = accuracy(Y[idx_train], labels[idx_train])
    loss_fn.backward()
    optimizer.step() # 参数更新
    # print("第{}次训练，loss损失值为：{}".format(epoch,loss_fn))
    loss_val = F.nll_loss(Y[idx_val], labels[idx_val])
    acc_val = accuracy(Y[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_fn.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_fn, gat_model.state_dict()


def compute_test(i,output_file='test_results.txt'):
    with open(output_file, 'w', encoding='utf-8') as f:
        gat_model.eval()

        output = gat_model(features, adj)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])

        acc_test = accuracy(output[idx_test], labels[idx_test])

        macro_f1 = calculate_macro_f1(output[idx_test], labels[idx_test])

        micro_f1 = calculate_micro_f1(output[idx_test], labels[idx_test])

        print("Test set results:",
                        "loss= {:.4f}".format(loss_test.data.item()),
                        "accuracy= {:.4f}".format(acc_test.data.item()),
                        "macro_f1= {:.4f}".format(macro_f1),
                        "micro_f1= {:.4f}".format(micro_f1),
                         "di--{}ci".format(i) )


        # 使用文件写入代替打印

        f.write("Test set results:\n")

        f.write(f"loss= {loss_test.item():.4f}\n")

        f.write(f"accuracy= {acc_test.item():.4f}\n")

        f.write(f"macro_f1= {macro_f1:.4f}\n")

        f.write(f"micro_f1= {micro_f1:.4f}\n\n")

        for i in range(NCLASS):
            precision = calculate_precision(output[idx_test], labels[idx_test], NCLASS)[i].item()

            recall = calculate_recall(output[idx_test], labels[idx_test], NCLASS)[i].item()

            division = precision + recall

            F1 = 0 if division == 0 else 2 * (precision * recall) / division

            print("第{}类的精确度为：{}".format(i, precision))
            print("第{}类的召回率为：{}".format(i, recall))
            print("第{}类的F1为：{}".format(i, F1))
            print("-----------------------------")

            # 使用文件写入代替打印

            f.write(f"第{i}类的精确度为：{precision}\n")

            f.write(f"第{i}类的召回率为：{recall}\n")

            f.write(f"第{i}类的F1为：{F1}\n")

            f.write("-----------------------------\n")

if __name__ == '__main__':

    num = 20
    for i in range(num):

       LOSS_ARR = []
       BEST = None
       MODEL_INFO = []

       for epoch in range(EPOCH):
           epoch_loss, param = train(epoch)

           if epoch_loss < BEST_LOSS:
               BEST_LOSS = epoch_loss
               BEST_MODEL_WTS = copy.deepcopy(param)

       torch.save(BEST_MODEL_WTS, 'best_model.pth')
       print("最佳训练结果：{}".format(BEST_LOSS))

       gat_model.load_state_dict(torch.load('best_model.pth', weights_only=True))

       compute_test(i,"./shenzhen-01/CiteSeer/wyxl/test_chameleon_{}.txt".format(i))



