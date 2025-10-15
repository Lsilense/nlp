from datasets import load_dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

# 加载 IMDb 数据集
dataset = load_dataset("imdb")

# 加载预训练的 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理函数：将文本转化为 BERT 可接受的输入格式
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# 对数据集进行预处理
train_dataset = dataset['train'].map(preprocess_function, batched=True)
test_dataset = dataset['test'].map(preprocess_function, batched=True)

# 转换为 PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)
