import argparse
import os
import yaml
from src.models.cnn.model_cnn import CNNModel
from src.utils.utils import load_data, save_model


def main(config):
    # Load training data
    train_data = load_data(config["data"]["train_file"])
    test_data = load_data(config["data"]["test_file"])

    # Initialize the CNN model
    model = CNNModel(config["model"])

    # Train the model
    model.train(train_data)

    # Evaluate the model
    accuracy = model.evaluate(test_data)
    print(f"Model accuracy: {accuracy:.2f}%")

    # Save the trained model
    save_model(model, config["model"]["save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_local.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
