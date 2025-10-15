#!/bin/bash

# Activate the virtual environment if needed
# source venv/bin/activate

# Set environment variables
export CONFIG_PATH="./config/config_local.yaml"

# Run the main script for the chosen model
# Uncomment the model you want to run

# For BERT model
# python src/models/bert/main.py

# For CNN model
# python src/models/cnn/main.py

echo "Model training and evaluation script executed."