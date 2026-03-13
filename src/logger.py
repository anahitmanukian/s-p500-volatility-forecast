import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging():
    config_path = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)