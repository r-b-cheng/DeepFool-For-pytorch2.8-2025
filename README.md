# DeepFool 项目说明

本项目基于 PyTorch 实现 DeepFool 对抗样本生成算法，并提供在 ImageNet 与 MNIST 上的示例、对抗数据集批量生成器、微调与鲁棒性评估脚本。

## 目录结构
- `Python/deepfool.py`：DeepFool 原始实现（L2 范数）
- `Python/deepfool_for_infinite_norm.py`：DeepFool 的无穷范数版本
- `Python/test_deepfool.py`：在 ImageNet 样例上演示对抗样本生成（ResNet34 预训练）
- `Python/moudle/LeNet.py`：MNIST 的 LeNet5 模型定义（权重位于 `moudle/lenet5_mnist.pth`） 准确率为98。87% 训练了20轮
- `Python/finetunning.py`：在 MNIST 上微调模型,以查看论文中多训练5轮后对模型鲁棒性的影响，最后的准确率为99.15%
- `Python/eval_benchmark_table1.py` / `Python/eval_benchmark_table2.py`：验证论文中表格1和表格2的第一行的结果
- `Python/utilis/adversarial_sample_generator.py`：按批数据逐张生成对抗样本并整合为批次的 DataLoader（默认 `batch_size=64`）

## 环境要求
- Python 3.x
- PyTorch2.8、torchvision
- numpy、Pillow、matplotlib


## 快速开始

### ImageNet 示例（单张图片）
```bash
python Python/test_deepfool.py
```
脚本会加载 `ResNet34` 预训练模型，输出原始分类与对抗分类，并展示生成的对抗图像。

### MNIST 模型微调
```bash
python Python/finetunning.py
```
微调完成后，权重会保存到 `Python/moudle/lenet5_mnist_finetuned.pth`。

## 生成对抗训练集（逐张处理、按批整合）

本项目提供了一个生成器，支持对一个 `DataLoader` 的每个批次逐张生成对抗样本，并最终返回一个新的 `DataLoader`（默认 `batch_size=64`）：

说明：
- 生成器会遍历每个批次，按图片逐张调用 DeepFool 计算扰动并收集结果；最后堆叠为 `TensorDataset` 并返回新的 `DataLoader`。

生成时长大约为1.5小时


## 参考
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard: DeepFool: a simple and accurate method to fool deep neural networks. CVPR 2016.
