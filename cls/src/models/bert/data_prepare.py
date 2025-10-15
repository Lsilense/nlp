import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer


def load_data(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None, names=["text", "label"])
    return data


def preprocess_data(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        data["text"].tolist(), padding=True, truncation=True, return_tensors="pt"
    )
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["label"])
    return inputs, labels


def prepare_data(train_file, test_file):
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    train_inputs, train_labels = preprocess_data(train_data)
    test_inputs, test_labels = preprocess_data(test_data)

    return (train_inputs, train_labels), (test_inputs, test_labels)
