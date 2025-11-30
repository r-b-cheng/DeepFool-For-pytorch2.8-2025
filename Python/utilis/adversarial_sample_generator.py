import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import collections
import deepfool
# from torch.autograd.gradcheck import zero_gradients 一个比较旧的API 已经被弃用，改为下面的实现
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

from torch.utils.data import TensorDataset, DataLoader

def generate_adversarial_dataset(dataset, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param dataset: PyTorch dataset (e.g., MNIST)
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: A PyTorch DataLoader containing adversarial examples with batch_size=64
    """
    adversarial_images = []
    adversarial_labels = []
    
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        # print("Using GPU")
        net = net.cuda()
    else:
        print("Using CPU")

    net.eval()

    for images, true_labels in dataset:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            true_labels = true_labels.unsqueeze(0)
        batch_size = images.size(0)
        for b in range(batch_size):
            image = images[b:b+1][0]
            true_label = true_labels[b]
            if is_cuda:
                image = image.cuda()
            f_image = net.forward(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
            I = (np.array(f_image)).flatten().argsort()[::-1]
            I = I[0:num_classes]
            label = I[0]

            input_shape = image.cpu().numpy().shape
            pert_image = copy.deepcopy(image)
            w = np.zeros(input_shape)
            r_tot = np.zeros(input_shape)

            loop_i = 0
            x = Variable(pert_image, requires_grad=True)
            fs = net.forward(x)
            fs_list = [fs[0,I[k]] for k in range(num_classes)]
            k_i = label

            while k_i == label and loop_i < max_iter:

                pert = np.inf
                fs[0, I[0]].backward(retain_graph=True)
                grad_orig = x.grad.data.cpu().numpy().copy()

                for k in range(1, num_classes):
                    zero_gradients(x)

                    fs[0, I[k]].backward(retain_graph=True)
                    cur_grad = x.grad.data.cpu().numpy().copy()

                    w_k = cur_grad - grad_orig
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                    pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                    if pert_k < pert:
                        pert = pert_k
                        w = w_k

                r_i =  (pert+1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)

                if is_cuda:
                    pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
                else:
                    pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

                x = Variable(pert_image, requires_grad=True)
                fs = net.forward(x)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())

                loop_i += 1

            adversarial_images.append(pert_image.cpu().detach())
            adversarial_labels.append(true_label.item() if isinstance(true_label, torch.Tensor) else true_label)
    # Convert lists to tensors
    # adversarial_images is a list of tensors with shape (C, H, W), we stack them to get (N, C, H, W)
    adv_images_tensor = torch.stack(adversarial_images)
    adv_labels_tensor = torch.tensor(adversarial_labels)

    # Create a TensorDataset
    adv_dataset = TensorDataset(adv_images_tensor, adv_labels_tensor)

    # Create a DataLoader
    adv_loader = DataLoader(adv_dataset, batch_size=64, shuffle=True)
    
    return adv_loader
