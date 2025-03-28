from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 0
learning_rate = 0.05
num_epochs = 10
batch_size = 8

# Architecture
num_features = 2
num_classes = 3

data = np.genfromtxt('../datasets/iris.data', delimiter=',', dtype=str)
X, y = data[:, [2, 3]], data[:, 4]
X = X.astype(float)  # convert string to float for X values. y values are already strings.

# 创建类别名称到数字的映射字典
d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# 将原始文本标签数组转换为对应的数字标签
y = np.array([d[x] for x in y])  # 遍历y中的每个类别名称，通过字典查找转换为数字
# 将数组类型转换为整型（注意：np.int在较新numpy版本中已弃用，建议改用int）
y = y.astype(np.int)

print('Class label counts:', np.bincount(y))
print('X.shape:', X.shape)
print('y.shape:', y.shape)

# shuffling & train/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# DataLoaders
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        training_example, training_label = self.X[index], self.y[index]
        return training_example, training_label

    def __len__(self):
        return self.y.shape[0]



train_dataset = MyDataset(X[:100], y[:100])
test_dataset = MyDataset(X[100:], y[100:])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,  # want to shuffle the dataset
                          num_workers=0)  # number processes/CPUs to use，4核

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)


# module
class SoftmaxRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, X):
        logits = self.linear(X)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = SoftmaxRegression(num_features=num_features, num_classes=num_classes)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

torch.manual_seed(random_seed)


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)  # 取probas的最大值并按列堆叠
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float() / num_examples * 100


for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):  # batch_idx从0开始，train_loader有多少个batch就有多少个batch_idx，features是一个batch的特征，targets是一个batch的标签。​
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)  # 交叉熵损失函数，logits是模型的原始输出，targets是真实标签。​
        optimizer.zero_grad()  # 梯度清零，避免上一次的梯度影响这一次的梯度计算。​
        cost.backward()  # 反向传播，计算梯度。​
        optimizer.step()  # 更新参数

        if not batch_idx % 50:  # 每50个batch打印一次信息，batch_idx从0开始，所以50的倍数就是50个batch。​
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'  # 打印当前的epoch数、总epoch数、当前的batch数、总batch数、当前的损失值。​
                  % (epoch + 1, num_epochs, batch_idx, len(train_dataset) // batch_size,
                     cost))  # 打印当前的epoch数、总epoch数、当前的batch数、总batch数、当前的损失值。

    with torch.set_grad_enabled(False):  # 关闭梯度计算，因为我们只关心准确率，不关心梯度。​
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
            epoch + 1, num_epochs,
            compute_accuracy(model, train_loader)))


print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))


# ​创建适配plot_decision_regions的ModelWrapper类

class ModelWrapper():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, X):
        features = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits, probas = self.model(features)
        _, predicted_labels = torch.max(probas, 1)

        return predicted_labels.cpu().numpy()  # 返回cpu上的numpy数组，因为plot_decision_regions需要numpy数组。​

mymodel = ModelWrapper(model, device=device)  # 创建ModelWrapper对象，用于适配plot_decision_regions函数。​

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X, y, mymodel)
plt.show()


