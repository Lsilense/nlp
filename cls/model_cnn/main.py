import torch.optim as optim
import torch.nn as nn
import time
import torch
from model_cnn import CNNModel
from data_prepare import trainloader, testloader

# 初始化模型
model = CNNModel().cuda()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD优化器

# 训练过程
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for inputs, labels in trainloader:
        inputs, labels = inputs.cuda(), labels.cuda()  # 将数据转移到 GPU

        optimizer.zero_grad()  # 清空梯度

        # 正向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 计算损失

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    # 训练结束后的统计信息
    end_time = time.time()
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}, Accuracy: {100 * correct/total:.2f}%, Time: {end_time - start_time:.2f}s"
    )
