import torch.nn as nn
from transformers import BertModel

class BertForSequenceClassification(nn.Module):
    def __init__(self):
        super(BertForSequenceClassification, self).__init__()
        
        # 使用预训练的 BERT 模型
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # 一个线性层进行分类
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 2 分类：正面、负面
        
    def forward(self, input_ids, attention_mask):
        # 获取 BERT 输出的最后一层隐状态
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取 [CLS] token 对应的隐状态作为句子的表示
        pooled_output = outputs.pooler_output
        
        # 通过分类层
        logits = self.classifier(pooled_output)
        return logits
