# 项目概述

这是一个使用PyTorch框架进行深度学习实践的代码库，主要包含以下几个核心组件：

1. **Fashion-MNIST数据集处理** (`fashionMNIST.py`)：用于加载和预处理Fashion-MNIST数据集，包括数据可视化和数据加载器的实现。
2. **Softmax回归模型** (`softmax.py`)：实现了基于Softmax回归的手写服装识别模型，包括模型定义、训练和评估过程。
3. **线性回归模型** (`linear_regression.py`)：实现了简单的线性回归模型，用于演示基本的机器学习概念。
4. **GPU支持** (`GPU.py`)：用于检测和配置GPU环境以加速模型训练。

## 主要技术栈

- PyTorch: 深度学习框架
- torchvision: 用于计算机视觉相关的工具和数据集
- d2l: 《动手学深度学习》书籍配套的工具库
- matplotlib: 用于数据可视化

## 开发约定

- 使用PyTorch进行模型定义和训练
- 使用torchvision处理图像数据集
- 使用d2l库中的工具函数简化代码实现
- 使用matplotlib进行结果可视化

## 构建和运行

由于这是一个Python深度学习项目，没有传统的构建步骤。直接运行Python文件即可：

```bash
# 运行线性回归示例
python linear_regression.py

# 运行Softmax回归示例
python softmax.py

# 查看Fashion-MNIST数据集处理
python fashionMNIST.py
```

## 数据集

项目使用Fashion-MNIST数据集，该数据集包含10类时尚物品的图像：
- t-shirt（T恤）
- trouser（裤子）
- pullover（套头衫）
- dress（连衣裙）
- coat（外套）
- sandal（凉鞋）
- shirt（衬衫）
- sneaker（运动鞋）
- bag（包）
- ankle boot（短靴）

数据集会自动下载并存储在`./data`目录中。