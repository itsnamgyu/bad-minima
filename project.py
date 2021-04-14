import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")

def get_weights_path(key):
    return os.path.join(WEIGHTS_DIR, key)
