import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def load_config_model(path):
    """
    Simple method for loading the config file (.json) of the model
    """
    full_path = os.path.join(ROOT_DIR, "data", "config", path)
    with open(full_path, "r") as f:
        config = json.load(f)
    return config
