import os

import torch
from scipy.io import loadmat
import random
import numpy as np
import matplotlib.pyplot as plt
import dataprocess
import cv2
import torch.nn.functional as F

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.hub import load_state_dict_from_url




# #加载预训练的AlexNet，设置参数为pretrained=True即可
# #如果只需要AlexNet的网络结构，设置参数为pretrained=False


# #因为summary默认在cuda上加载模型所以要把模型加载到GPU上

# # 创建一个包含复数的张量
# complex_tensor = torch.tensor([3+4j, 1+2j, 2+2j])

# # 计算复数张量的范数
# norm = torch.linalg.norm(complex_tensor, ord=2, dim=-1, keepdim=True)

# # 对复数张量进行归一化
# normalized_complex = complex_tensor / norm

# print(normalized_complex)

print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(1))    # 根据索引号得到G
print(torch.__version__)
