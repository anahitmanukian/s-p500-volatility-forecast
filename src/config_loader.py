import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_processed_data_path(config=None):
    """Get processed data path from config"""
    if config is None:
        config = load_config()
    return config['paths']['processed_data']

def get_raw_data_path(config=None):
    """Get raw data path from config"""
    if config is None:
        config = load_config()
    return config['paths']['raw_data']