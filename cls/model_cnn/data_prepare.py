import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据预处理：归一化到[0,1]区间，并进行数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),               # 将图片转换为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 下载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = DataLoader(testset, batch_size=64, shuffle=False)
