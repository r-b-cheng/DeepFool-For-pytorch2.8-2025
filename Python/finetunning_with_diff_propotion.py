# this file is designed to see whether the training on the adversial examples can improve the robustness of the model
'''预期的结果：
    1. 模型在测试集上的准确率上升
    2. 样本的平均扰动上升
'''
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from deepfool import deepfool
from moudle import LeNet
from utilis import evaluate_the_robustness

def finetune_with_proportion(proportion, epochs=5, base_lr=0.001, momentum=0.9, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader_eval = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    test_loader_acc = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet.LeNet5()
    net.load_state_dict(torch.load('./moudle/lenet5_mnist.pth',map_location=device))
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=base_lr*0.5, momentum=momentum)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if proportion > 0:
                k = int(images.size(0) * proportion)
                if k > 0:
                    net.eval()
                    adv_list = []
                    for j in range(k):
                        r, loop_i, label_orig, label_pert, pert_image = deepfool(images[j:j+1].detach().cpu(), net)
                        adv_list.append(pert_image)
                    net.train()
                    adv_images = torch.cat(adv_list, dim=0).to(device)
                    images[:k] = adv_images

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    torch.save(net.state_dict(), f'./moudle/lenet5_mnist_finetuned_p{proportion}.pth')

    robustness = evaluate_the_robustness(net, test_loader_eval)
    total = 0
    correct = 0
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader_acc:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f'proportion={proportion}, robustness={robustness}, acc={acc}')


if __name__ == '__main__':
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        finetune_with_proportion(p)
