import os
from contextlib import contextmanager


@contextmanager
def change_working_dir(new_dir):
    current_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(current_dir)
