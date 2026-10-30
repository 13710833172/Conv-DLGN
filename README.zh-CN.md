# Conv-DLGN：卷积可微逻辑门网络

[English](README.md) | **简体中文**

**以可导出逻辑树构建空间特征层级。**

本项目是 [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732) 的独立 PyTorch 实现与复现研究。下述贡献对应本仓库提供的实现与评估；卷积逻辑树、OR 池化与残差初始化遵循原论文的方法。

## 摘要

使用逻辑门完成图像分类，需要同时建立局部特征提取机制与从可微训练到离散执行的转换路径。Conv-DLGN 通过卷积逻辑树、OR 池化与稀疏逻辑分类头，构建保留空间结构的逻辑计算层级。训练时通过可学习的运算混合进行梯度优化，导出时在保留连接结构的同时，为每个门指定一种布尔运算。代表性 CIFAR-10 B 配置取得 **70.32%（soft）与 70.27%（hard）测试准确率**，两种模式相差 **0.05 个百分点**。该结果直接展示了导出逻辑树模型的分类表现，以及 soft 与 hard 执行之间的接近程度。

## 贡献

- **保留空间结构的逻辑计算。** 局部深度为 3 的逻辑树与 OR 池化，在卷积层级中保留图像特征图。
- **完整的 soft-to-hard 工作流。** 将运算学习、残差门初始化、验证集检查点选择与离散导出整合进 PyTorch 实现。
- **直接检验导出模型。** 在 MNIST 与 CIFAR-10 上配对报告 soft/hard 结果，其中 CIFAR-10 B 配置的 hard 准确率达到 70.27%。

## 方法概述

![Conv-DLGN 方法：图像编码、局部逻辑树卷积、OR 池化、稀疏逻辑分类头与类别分组读出；单棵树含八个叶子与七个门。](docs/figures/overview.svg)

*图 1：局部逻辑树在不同图像位置共享计算，输出空间特征图，再经池化与稀疏逻辑头完成分类。插图展示单个输出通道对应的八叶子、七逻辑门结构。*

**局部组合。** 输入通过阈值或位平面编码生成特征通道。每棵卷积树从局部感受野中采样八个固定叶子，涉及至多两个输入通道，再通过三层、共七个双输入门生成输出。同一棵可学习逻辑树作用于不同空间位置。OR 池化采用 max t-conorm，逻辑头通过类别分组求和形成分类分数。

**运算学习与离散导出。** 每个门学习 16 种布尔运算的 softmax 加权混合；残差初始化偏向恒等运算。导出时选择 argmax 对应运算，保留树叶子与稀疏分类头的连接。实现见[卷积逻辑树](src/models/conv_logic_tree_layer.py)、[网络组合](src/models/logic_tree_net.py)与[hard 模型](src/models/hard_logic_tree_net.py)。

## 实验

**研究问题：空间逻辑树分类器在门离散化后，能否保持分类表现？** 表 1 在每个配置内部配对比较 soft 与 hard 模式。

| 数据集 / 配置 | Soft 测试准确率 | Hard 测试准确率 | Soft − hard（百分点） |
| --- | ---: | ---: | ---: |
| MNIST / S | 98.49% | 97.55% | 0.94 |
| CIFAR-10 / S | 58.93% | 57.20% | 1.73 |
| CIFAR-10 / B | **70.32%** | **70.27%** | **0.05** |

*表 1：已完成的代表性本地配置。每行比较同一选定模型在门导出前后的表现；不同行的模型配置与训练设置不同。*

**核心发现。** CIFAR-10 B 是主要展示结果：导出模型取得 70.27% 准确率，与 soft 执行相差 0.05 个百分点。MNIST S 与 CIFAR-10 S 提供了同一实现路径下的其他工作点。跨行差异同时涉及模型与训练选择，因此表格用于论证配对转换表现，不作为受控的规模消融实验。

**实验口径。** 按验证集 soft 准确率选择检查点，再评估测试集。各行是单个已完成配置的结果，不是多个独立随机种子的均值。输入编码与教师模型使用由配置决定；启用教师的 B/L/G 配置在训练时需要对应的外部教师检查点。仓库展示 PyTorch 中的分类与导出测量；吞吐与能耗需要在执行后端上独立测量。数据集、检查点与原始实验历史保留在仓库之外。

## 快速使用

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 完整训练。
python src/train.py --config configs/paper_table6/mnist_s.yaml
python src/train.py --config configs/paper_table6/cifar10_s.yaml
```

示例启动新的 S 配置训练。使用实际生成的检查点评估或导出：

```bash
python src/evaluate.py --checkpoint path/to/checkpoint.pt
python src/discretize.py --checkpoint path/to/checkpoint.pt
python -m pytest tests -v
```

检查点保存在各次运行目录中。表格汇总此前完成的配置，上述示例命令不包含历史实验环境的全部设置。

## 仓库导航

| 入口 | 内容 |
| --- | --- |
| [configs/paper_table6/](configs/paper_table6/) | MNIST S/M/L 与 CIFAR-10 S/M/B/L/G 配置；已展示的实验见表 1 |
| [src/data/](src/data/) | 阈值与位平面输入编码 |
| [src/models/](src/models/) | Soft/hard 逻辑树、OR 池化与分类头 |
| [MODEL_LAYER_SUMMARY.md](MODEL_LAYER_SUMMARY.md) | 中文网络层级说明 |
| [REFERENCE_NOTES.md](REFERENCE_NOTES.md) | 方法来源与实现范围 |

关联项目：[DLGN](https://github.com/13710833172/DLGN) 提供展平输入的稀疏逻辑网络；[Logic-SNN](https://github.com/13710833172/Logic-SNN) 引入脉冲时间动力学。

## 结论

Conv-DLGN 将空间特征提取连接到离散逻辑树执行。代表性 CIFAR-10 B 模型以 **70.27% hard 准确率、0.05 个百分点的 soft-to-hard 差值**，给出这一连接的直接证据。
