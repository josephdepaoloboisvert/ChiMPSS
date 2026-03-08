import os, glob


def ensure_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return True

