import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # 第一层卷积：输入3通道，输出32通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 第二层卷积：输入32通道，输出64通道，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 第三层全连接层：输入64*8*8维，输出512维
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # 第二层全连接层：输入512维，输出10维（CIFAR-10有10类）
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 卷积层 + 激活函数 + 池化层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 2x2最大池化

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # 展平处理，准备进入全连接层
        x = x.view(-1, 64 * 8 * 8)

        # 全连接层 + 激活函数
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
