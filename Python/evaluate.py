# this file is designed to see whether the training on the adversial examples can improve the robustness of the model
'''预期的结果：
    1. 模型在测试集上的准确率上升
    2. 样本的平均扰动上升
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
from moudle import LeNet
from utilis import robustness

if __name__ == '__main__':
    # load test data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet.LeNet5()
    net.to(device)
    net.load_state_dict(torch.load('./moudle/lenet5_mnist_finetuned.pth', map_location=device))
    '''
    # 在测试集上计算平均扰动
    robustness = robustness.evaluate_the_robustness(net, test_loader)
    print("The robustness of the finetuned model is: ", robustness)
    '''
    #测试在测试集上的准确率
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        image, label = data
        image = image.to(device)
        label = label.to(device)
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)
    acc = correct / total
    print(f"The accuracy of the finetuned model is:{acc*100}%")




