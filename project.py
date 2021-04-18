import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
HISTORIES_DIR = os.path.join(PROJECT_DIR, "histories")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")


def get_weights_path(key):
    return os.path.join(WEIGHTS_DIR, key)


def get_datasets_path(key):
    return os.path.join(DATASETS_DIR, key)


def get_histories_path(key):
    return os.path.join(HISTORIES_DIR, key)


def get_plots_path(key):
    return os.path.join(PLOTS_DIR, key)
