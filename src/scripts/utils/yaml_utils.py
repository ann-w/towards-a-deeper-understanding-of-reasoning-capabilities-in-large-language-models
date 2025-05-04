import io
import logging
import os

from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)


def load_experiment_settings(file_path: str) -> dict:
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return yaml.load(file)
    else:
        logging.info(f"File not found: {file_path}, using default settings.")
        return {}


def log_experiment_settings(experiment_settings):
    stream = io.StringIO()
    yaml.dump(experiment_settings, stream)
    logging.info(stream.getvalue())
