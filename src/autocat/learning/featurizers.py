import qml

import tempfile
import os
import numpy as np
import json

from typing import List

from ase import Atoms


def get_X(
    structures: List[Atoms],
    size: int = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    **kwargs,
):
    """
    Generate representation matrix X from list of ase.Atoms objects

    Parameters
    ----------

    structures:
        List of ase.Atoms objects to be used to construct X matrix

    size:
        Size of the largest structure to be supported by the representation.
        Default: number of atoms in largest structure within `structures`

    write_to_disk:
        Boolean specifying whether X should be written to disk as a json.
        Defaults to False.

    write_location:
        String with the location where X should be written to disk.

    Returns
    -------

    X:
        np.ndarray of X representation
    """
    if size is None:
        size = max([len(s) for s in structures])

    qml_mols = [ase_atoms_to_qml_compound(m) for m in structures]

    for mol in qml_mols:
        mol.generate_coulomb_matrix(size=size, **kwargs)

    X = np.array([mol.representation for mol in qml_mols])

    if write_to_disk:
        write_path = os.path.join(write_location, "X.json")
        X_list = X.tolist()
        with open(write_path, "w") as f:
            json.dump(X_list, f)
        print(f"X written to {write_path}")

    return X


def ase_atoms_to_qml_compound(ase_atoms: Atoms):
    """
    Converter from `ase.Atoms` to `qml.Compound`

    Parameters
    ----------

    ase_atoms:
        Atoms object to be converted to a qml.Compound

    Returns
    -------

    qml_compound:
        qml.Compound object corresponding to give Atoms object
    """
    with tempfile.TemporaryDirectory() as _tmp_dir:
        ase_atoms.write(os.path.join(_tmp_dir, "tmp.xyz"))
        qml_compound = qml.Compound(xyz=os.path.join(_tmp_dir, "tmp.xyz"))
        return qml_compound
