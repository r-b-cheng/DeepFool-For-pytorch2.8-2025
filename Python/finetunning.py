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
from utilis import robustness, adversarial_sample_generator

if __name__ == '__main__':
    # load train data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # load test data
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet.LeNet5()
    net.load_state_dict(torch.load('./moudle/lenet5_mnist.pth', map_location=device))
    net.to(device)
    net.eval()

    #avg_robustness1 = robustness.evaluate_the_robustness(net, test_loader)
    #print("The robustness before finetunning: ", avg_robustness1)

    print("Start finetunning...")
    # 先对于所有的训练集，计算对抗样本
    '''
    adv_train_data = []
    for i, data in enumerate(train_loader):
        image, label = data
        r, loop_i, label_orig, label_pert, pert_image = deepfool(image, net)
        adv_train_data.append((pert_image, label))
    '''
    adv_train_data = adversarial_sample_generator.generate_adversarial_dataset(train_loader, net)
    
    # 再训练5轮，用之前50%的学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001*0.5, momentum=0.9)
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(adv_train_data):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = net(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(adv_train_data)}")

    # 保存模型
    torch.save(net.state_dict(), './moudle/lenet5_mnist_finetuned.pth')

    # 在测试集上计算平均扰动
    avg_robustness2 = robustness.evaluate_the_robustness(net, test_loader)
    print("The robustness of the finetuned model is: ", avg_robustness2)
'''
    # 测试在测试集上的准确率
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
'''
    
