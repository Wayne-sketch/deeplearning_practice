'''
Author: ctx cuitongxin201024@163.com
Date: 2025-10-15 00:14:44
LastEditors: ctx cuitongxin201024@163.com
LastEditTime: 2025-10-15 22:37:59
FilePath: \deeplearning_practice\linear_regression.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, num_examples=1000)

# print(true_w)
# print(true_b)
# print(features.size(), labels.size())

# pytorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 一轮num_examples个样本，每个batch有batch_size个样本，一共num_examples/batch_size个batch
# 一轮1000个样本，每个batch有100个样本，一共10个batch

batch_size = 100
data_iter = load_array((features, labels), batch_size)

# next(iter(data_iter))

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))
# 访问线性层 初始化
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 均方误差 L2范数 默认reduction='mean'：L2范数平方取均值 reduction='sum'L2范数平方
loss = nn.MSELoss()
# SGD 随机梯度下降 实际是小批量梯度下降，而非单样本SGD BatchSGD和Mini-batchSGD都是单个样本loss对参数求梯度求和取平均值 等价于loss对参数求梯度
# 其他可选参数：
# momentum ：动量
# weight_decay ：L2 正则化
# dampening、nesterov 等
trainer = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练
# 每轮10次梯度更新，一共num_epochs轮
num_epochs = 500
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        # 梯度清零
        trainer.zero_grad()
        l.backward()
        # 模型更新
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


# batch_size调大，学习率需要适当缩小，提高训练轮数
# 大batch每次迭代梯度更接近全局梯度，更新方向更稳定，但缺少随机性
