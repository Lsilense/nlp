import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from src.models.bert.data_prepare import prepare_data
from config.config import Config

def main():
    # Load configuration
    config = Config()
    
    # Prepare data
    train_data, test_data = prepare_data(config.data_path)
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    model = BertForSequenceClassification.from_pretrained(config.bert_model, num_labels=config.num_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    model.train()
    for epoch in range(config.epochs):
        for batch in train_loader:
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = batch['label'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
    
    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(batch['label'].cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()