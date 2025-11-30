# eval the fist entry in the table 2

from moudle import LeNet
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
from deepfool_for_infinite_norm import deepfool_for_infinite_norm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = LeNet.LeNet5()
net.to(device)
net.load_state_dict(torch.load('./moudle/lenet5_mnist.pth',map_location=device))

net.eval()
# load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
# 对于整个数据集计算扰动的大小
perturb = 0.0
count = 0
acc = 0
for i, data in enumerate(test_loader):
    image, label = data
    image = image.to(device)
    label = label.to(device)
    out = net(image)
    _, pred = torch.max(out, 1)
    r, loop_i, label_orig, label_pert, pert_image = deepfool_for_infinite_norm(image, net)
    print("Original label = ", label_orig.item(), "Perturbed label = ", label_pert.item())

    current_perturb = np.linalg.norm(r.flatten()) / np.linalg.norm(image.cpu().numpy().flatten())
    perturb += current_perturb
    print("The perturbation is: ", current_perturb)
    count += 1

# 最后除以整个训练集的大小
print("The average perturbation is: ", perturb / count)

