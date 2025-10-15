def load_data(file_path):
    """Load data from a given file path."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return [line.strip() for line in data]

def save_data(data, file_path):
    """Save data to a given file path."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(f"{line}\n")

def evaluate_model(predictions, labels):
    """Evaluate the model's predictions against the true labels."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(labels) if labels else 0
    return accuracy

def preprocess_text(text):
    """Preprocess the input text for model compatibility."""
    # Example preprocessing steps
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text.strip()