import os.path


def get_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
