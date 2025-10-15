import os
import yaml

class Config:
    def __init__(self, config_file='config/config_local.yaml'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as file:
                return yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")

    def get(self, key, default=None):
        return self.config.get(key, default)

# Example usage
if __name__ == "__main__":
    config = Config()
    print(config.get('some_key', 'default_value'))  # Replace 'some_key' with actual key from your config file.