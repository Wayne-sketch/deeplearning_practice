'''
Author: ctx cuitongxin201024@163.com
Date: 2025-10-15 23:45:01
LastEditors: ctx cuitongxin201024@163.com
LastEditTime: 2025-10-17 01:28:35
FilePath: \deeplearning_practice\softmax.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from IPython import display
from d2l import torch as d2l
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 256

trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
mnist_test = datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax操作
# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X.sum(0, keepdim=True))
# print(X.sum(1, keepdim=True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim= True) # partation 2,1
    return X_exp / partition # 2,3 / 2,1 2,1扩展到2,3

# X = torch.normal(0, 1, (2, 5)) # 2,5
# X_prob = softmax(X) # 2,5
# print(X_prob)
# print(X_prob.sum(1)) #2,1 default keepdim=False

#softmax回归模型
def net(X):
    #before reshape 256 * 28 * 28 after reshape 256 * 784
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

# example 2 sample 3 calsses
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# 通过样本真实值分类标号 拿到对应标号预测值
# print(y_hat[[0, 1], y])

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

print(cross_entropy(y_hat, y))

# 预测类别与真实y元素进行比较
# y_hat batch_size * num_classes
def accuracy(y_hat, y):
# 计算预测正确的数量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 找出每行最大值所在下标 所在下标表示预测类别就是该下标
        y_hat = y_hat.argmax(axis = 1)
    # cmp是bool tensor y与y_hat类型一致才能比较
    cmp = y_hat.type(y.dtype) == y
    # 预测正确的数量 再转成浮点数 方便后续打印计算
    return float(cmp.type(y.dtype).sum())

print(accuracy(y_hat, y) / len(y))

def evaluate_accuracy(net, data_iter):
    # 计算在数据集上模型的准确率
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 手动实现的累加器小工具
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# 先评估随机出来的模型和迭代器
print(evaluate_accuracy(net, test_iter))

# 训练 迭代一次
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:  #@save
    """在动画中绘制数据（适用于脚本）"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid(True)

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.pause(0.001)  # 暂停以更新图像

    def show(self):
        plt.show()

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    animator.show()



lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):
    """预测标签并显示图片"""
    # 取一个batch
    for X, y in test_iter:
        break

    # 模型预测
    preds = net(X).argmax(axis=1)

    # 获取真实标签与预测标签的文本
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(preds)
    titles = [t + '\n' + p for t, p in zip(trues, preds)]

    # 绘图
    X = X[0:n].reshape((n, 28, 28))
    _, axes = plt.subplots(1, n, figsize=(12, 12))
    for i, (ax, img, title) in enumerate(zip(axes, X, titles)):
        ax.imshow(img.numpy(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

predict_ch3(net, test_iter)