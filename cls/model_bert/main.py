from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from model_bert import BertForSequenceClassification
from data_prepare import train_loader, test_loader
import torch.nn as nn
import torch

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练过程
def train(model, train_loader, optimizer, criterion, device):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0
    correct_preds = 0
    total_preds = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss, accuracy

# 训练模型
num_epochs = 3

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

def evaluate(model, test_loader, device):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 取出预测结果
            _, predicted = torch.max(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 测试模型
test_accuracy = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
