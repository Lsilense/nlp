import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from a text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    data = pd.read_csv(file_path, sep='\t', header=None)
    return data

def preprocess_data(data):
    """Preprocess the data for CNN model."""
    # Example preprocessing steps
    data.columns = ['text', 'label']
    data['text'] = data['text'].str.lower()  # Convert to lowercase
    data['text'] = data['text'].str.replace(r'\d+', '')  # Remove digits
    return data

def prepare_data(train_file, test_file):
    """Prepare training and testing data."""
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    return train_data, test_data

if __name__ == "__main__":
    train_file_path = os.path.join('..', '..', 'data', 'train.txt')
    test_file_path = os.path.join('..', '..', 'data', 'test.txt')
    
    train_data, test_data = prepare_data(train_file_path, test_file_path)
    print("Data preparation completed.")