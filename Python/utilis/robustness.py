import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
# 获取当前文件所在目录的上级目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

# 添加到系统路径
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
from deepfool import deepfool

def evaluate_the_robustness(net, data_loader, show=False, max_samples=None):
    net.eval()
    perturb = 0.0
    count = 0
    for i, (image, label) in enumerate(data_loader):
        if max_samples is not None and count >= max_samples:
            break
        r, loop_i, label_orig, label_pert, pert_image = deepfool(image, net)
        if show:
            plt.imshow(pert_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
            plt.show()
        print("Original label = ", label_orig, "Perturbed label = ", label_pert)
        perturb += np.linalg.norm(r.flatten()) / np.linalg.norm(image.cpu().numpy().flatten())
        count += 1
    return perturb / count if count > 0 else 0.0
