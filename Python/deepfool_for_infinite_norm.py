import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import collections
# from torch.autograd.gradcheck import zero_gradients 一个比较旧的API 已经被弃用，改为下面的实现
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def deepfool_for_infinite_norm(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()


    f_image = net.forward(Variable(image, requires_grad=False)).data.cpu().numpy().flatten()

    # f_image = net.forward(Variable(image[None,:,:,:], requires_grad=True)).data.cpu().numpy().flatten()
    # 得到网络的输出结果 并展平为一维数组
    I = (np.array(f_image)).flatten().argsort()[::-1]
    # 获取前num_classes个概率最大的类别的索引
    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    x = Variable(pert_image, requires_grad=True)
    # x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf  # 初始化扰动值 将扰动值设置成无穷大
        fs[0, I[0]].backward(retain_graph=True) # 对网络输出的最大类别（原始预测类别）执行反向传播
        # 表示是对第0个维度中索引最大的那个
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)  # 对其他类别的输出进行反向传播计算梯度
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            # 计算一范数
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten(), ord=1)

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * np.sign(w) 
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot #在最终的扰动计算中，通过 (1+overshoot) 的系数来稍微放大扰动，确保能够成功欺骗分类器

    return r_tot, loop_i, label, k_i, pert_image
