import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

class ModelBERT(nn.Module):
    def __init__(self, config):
        super(ModelBERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model'])
        self.dropout = nn.Dropout(config.get('dropout', 0.3))
        self.classifier = nn.Linear(self.bert.config.hidden_size, config['num_classes'])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        out = self.dropout(pooled_output)
        logits = self.classifier(out)
        return logits

    def train_model(self, train_loader, val_loader, device, epochs=3, lr=2e-5):
        self.to(device)
        optimizer = AdamW(self.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
            val_acc = self.evaluate(val_loader, device)
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")

    def evaluate(self, data_loader, device):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0

    def predict(self, input_ids, attention_mask, device):
        self.eval()
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = self(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
        return preds.cpu().numpy()

def load_data(config):
    # 实现数据加载和DataLoader的构建
    # 返回train_loader, val_loader, test_loader
    pass

if __name__ == "__main__":
    import config.config as config_module

    config = config_module.load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBERT(config)

    # 加载数据
    train_loader, val_loader, test_loader = load_data(config)
    model.train_model(train_loader, val_loader, device, epochs=config.get('epochs', 3), lr=config.get('lr', 2e-5))
    test_acc = model.evaluate(test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")