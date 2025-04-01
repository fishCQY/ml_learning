# @file    mlp_fromscratch_sigmoid_mse.py
# @author  cqy 3049623863@qq.com
# @date    2025/03/31 22:13:13
# @version 1.0
# @brief

# scratch using
# - sigmoid activation in the hidden layer
# - sigmoid activation in the output layer
# - Mean Squared Error loss function


import matplotlib.pyplot as plt
import pandas as pd
import torch

import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True  # 设置cuDNN后端使用确定性算法，每次运行都会得到相同的计算结果

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
num_epochs = 50
batch_size = 100

train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


class MultiLayerPerceptron():
    def __init__(self, num_features, num_hidden, num_classes):
        super(MultiLayerPerceptron, self).__init__()

        self.num_classes = num_classes
        #使用均值为0，标准差为0.1的正态分布初始化权重矩阵和偏置向量，normal_1()可以修改原始的张量值
        self.weight_1 = torch.zeros(num_hidden, num_features, dtype=torch.float).normal_(0.0, 0.1)
        self.bias_1 = torch.zeros(num_hidden, dtype=torch.float)

        self.weight_o = torch.zeros(self.num_classes, num_hidden, dtype=torch.float).normal_(0.0, 0.1)
        self.bias_o = torch.zeros(self.num_classes, dtype=torch.float)

    def forward(self, x):
        # input dim:  [n_examples,n_features],[n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_1 = torch.mm(x, self.weight_1.t()) + self.bias_1
        a_1 = torch.sigmoid(z_1)

        # input dim:  [n_examples,n_hidden],[n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_2 = torch.mm(a_1, self.weight_o.t()) + self.bias_o
        a_2 = torch.sigmoid(z_2)

        return a_1, a_2 # 返回的是第一层的输出和第二层的输出

    def backward(self, x, a_1, a_2, y):
        # one-hot encoding
        # y:num_examples,correct_classlabel
        # 创建一个大小为 (y.size(0), self.num_classes) 的全零张量，每个值为32位浮点数
        y_onehot = torch.FloatTensor(y.size(0), self.num_classes)
        y_onehot.zero_()
        # dim=1，每一行（对应一个样本）的某个列位置会被设置为 1
        # y.view(-1, 1)：将标签张量 y 转换为形状为 (batch_size, 1) 的二维张量。
        # 其中，-1 表示自动计算该维度的大小，使得总元素数保持不变。
        # .long()：将索引转换为长整型（torch.LongTensor），这是 scatter_ 方法的格式要求。
        # 在 index 指定的位置填充值 1
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        # a_2 对应类别的概率
        # input/output dim: [n_examples, n_classes]
        # 用的是MSE损失，导数是2*(预测值-真实值)，然后除以样本数量取平均
        dloss_da2 = 2. * (a_2 - y_onehot) / y.size(0)

        # input/output dim: [n_examples, n_classes]
        # sigmoid函数的导数是sigmoid(x)*(1-sigmoid(x))
        da2_dz2 = a_2 * (1. - a_2)  # sigmoid derivative

        # input/output dim: [n_examples, n_classes]
        delta_out = dloss_da2 * da2_dz2  # "delta (rule) placeholder"

        # gradient for output weights
        # [n_examples, n_hidden] 
        dz2_dw_out = a_1

        # input dim: [n_classlabels, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classlabels, n_hidden]
        dloss__dw_out = torch.mm(delta_out.t(), dz2_dw_out)
        dloss__db_out = torch.sum(delta_out, dim=0)  # 按列求和

        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight

        # [n_classes, n_hidden]
        dz2_da1 = self.weight_o

        # output dim: [n_examples, n_hidden]
        dloss_a1 = torch.mm(delta_out, dz2_da1)

        # [n_examples, n_hidden]
        da1_dz1 = a_1 * (1. - a_1)  # sigmoid derivative

        # [n_examples, n_features]
        dz1_dw1 = x
        # output dim: [n_hidden, n_features]
        dloss_dw1 = torch.mm((dloss_a1 * da1_dz1).t(), dz1_dw1)
        dloss_db1 = torch.sum((dloss_a1 * da1_dz1), dim=0)

        return dloss__dw_out, dloss__db_out, dloss_dw1, dloss_db1
# one-hot encoding
def to_onehot(y, num_classes):
    y_onehot = torch.FloatTensor(y.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot


def loss_func(targets_onehot, probas_onehot):
    return torch.mean(torch.mean((targets_onehot - probas_onehot) ** 2, dim=0))


def compute_mse(net, data_loader):
    curr_mse, num_examples = torch.zeros(model.num_classes).float(), 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28 * 28) # 图片个数，28*28
            logits, probas = net.forward(features)
            y_onehot = to_onehot(targets, model.num_classes)
            loss = torch.sum((y_onehot - probas) ** 2, dim=0)
            num_examples += targets.size(0)
            curr_mse += loss

        curr_mse = torch.mean(curr_mse / num_examples, dim=0)
        return curr_mse #[1,num_classes]


def train(model, data_loader, num_epochs, learning_rate=0.1):
    minibatch_cost = []
    epoch_cost = []

    for e in range(num_epochs):
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.view(-1, 28 * 28)

            a_1, a_2 = model.forward(features)

            dloss__dw_out, dloss__db_out, dloss_dw1, dloss_db1 = \
                model.backward(features, a_1, a_2, targets)
            model.weight_1 -= learning_rate * dloss_dw1
            model.bias_1 -= learning_rate * dloss_db1
            model.weight_o -= learning_rate * dloss__dw_out
            model.bias_o -= learning_rate * dloss__db_out

            curr_cost = loss_func(to_onehot(targets, model.num_classes), a_2)
            minibatch_cost.append(curr_cost)
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                      % (e + 1, num_epochs, batch_idx, len(train_loader), curr_cost))

        curr_cost = compute_mse(model, train_loader)
        epoch_cost.append(curr_cost)
        print('Epoch: %03d/%03d |' % (e + 1, num_epochs), end="")
        print(' Train MSE: %.5f' % curr_cost)

    return minibatch_cost, epoch_cost


# training
torch.manual_seed(random_seed)
model = MultiLayerPerceptron(num_features=28 * 28, num_hidden=50, num_classes=10)

minibatch_cost, epoch_cost = train(model,
                                   train_loader,
                                   num_epochs=num_epochs,
                                   learning_rate=0.1)

# minibatch_cost记录了每个batch的损失
plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Mean Squared Error')
plt.xlabel('Minibatch')
plt.show()
# 记录了每个epoch的损失
plt.plot(range(len(epoch_cost)), epoch_cost)
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.show()


def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28 * 28)
            _, outputs = net.forward(features)
            predicted_labels = torch.argmax(outputs, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

        return correct_pred.float() / num_examples * 100


print('training accuracy: %.2f' % compute_accuracy(model, train_loader))
print('test accuracy: %.2f' % compute_accuracy(model, test_loader))

import matplotlib
import matplotlib.pyplot as plt

for features, targets in test_loader:
    break

fig, ax = plt.subplots(1, 4)
for i in range(4):
     # 1. 形状调整：将第i个样本的展平向量 [784] 还原为 [28, 28] 的二维图像
    # cmap=matplotlib.cm.binary  # 2. 颜色映射：使用黑白二值色阶（0=白，1=黑）
    ax[i].imshow(features[i].view(28, 28), cmap=matplotlib.cm.binary)

plt.show()
# 这里传入数据集中前四张图片进行分类
_, predictions = model.forward(features[:4].view(-1, 28 * 28))
# 对每个样本的输出概率，取概率最大的类别索引作为预测结果，取每一行中概率最大的
predictions = torch.argmax(predictions, dim=1)
print('predicted labels', predictions)
