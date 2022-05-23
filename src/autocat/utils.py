import os
from contextlib import contextmanager
from ase import Atoms


@contextmanager
def change_working_dir(new_dir: str):
    current_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(current_dir)


def flatten_structures_dict(autocat_dict: dict):
    structure_list = []
    for element in autocat_dict:
        if isinstance(autocat_dict[element], dict):
            structure_list.extend(flatten_structures_dict(autocat_dict[element]))
        elif isinstance(autocat_dict[element], Atoms):
            structure_list.append(autocat_dict[element])
    return structure_list
