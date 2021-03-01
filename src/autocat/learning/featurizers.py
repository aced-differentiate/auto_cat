import qml
from matminer.featurizers.site import GaussianSymmFunc
from matminer.featurizers.site import SOAP

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import EwaldSumMatrix
from dscribe.descriptors import CoulombMatrix

import tempfile
import os
import numpy as np
import json

from typing import List

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from autocat.io.qml import ase_atoms_to_qml_compound


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


def adsorbate_featurization(
    structure: Atoms,
    adsorbate_indices: List[int],
    featurizer: str = "behler-parinello",
    **kwargs,
):
    """
    Featurizes adsorbate on a support via `matminer.featurizers.site`
    functions.

    Parameters
    ----------

    structure:
        Atoms object of full structure (adsorbate + slab) to be featurized

    adsorbate_indices:
        List of atomic indices specifying the adsorbate to be
        featurized

    featurizer:
        String indicating featurizer to be used.

        Options:
        behler-parinello (default): gaussian symmetry functions
        soap: smooth overlap of atomic positions

    Returns
    -------

    representation:
        Np.ndarray of adsorbate representation

    """
    conv = AseAtomsAdaptor()
    pymat_struct = conv.get_structure(structure)

    representation = np.zeros(len(adsorbate_indices))

    if featurizer == "behler-parinello":
        feat = GaussianSymmFunc(**kwargs)

    elif featurizer == "soap":
        feat = SOAP(**kwargs)

    else:
        raise NotImplementedError("selected featurizer not implemented")

    for i, idx in enumerate(adsorbate_indices):
        representation[i] = feat.featurize(pymat_struct, idx)

    return representation


def full_structure_featurization(
    structure: Atoms,
    size: int = None,
    featurizer: str = "sine_matrix",
    permutation: str = "none",
    n_jobs: int = 1,
    **kwargs,
):
    """
    Featurizes the entire structure (including the adsorbate) using
    representations available in `dscribe` and `qml`

    Parameters
    ----------

    structure:
        Atoms object of structure to be featurized

    size:
        Size of the largest structure to be supported by the representation.
        Default: number of atoms in `structures`

    featurizer:
        String indicating featurizer to be used.

        Options:
        - sine_matrix (default)
        - coulomb_matrix (N.B.: does not support periodicity)
        - bag_of_bonds

    permutation:
        String specifying how ordering is handled. This is fed into
        `dscribe` featurizers (ie. sine_matrix, ewald_sum_matrix, coulomb_matrix)
        Default: "none", maintains same ordering as input Atoms structure
        (N.B. this differs from the `dscribe` default)

    n_jobs:
        Int specifiying number of parallel jobs to run which is fed into `dscribe`
        featurizers (ie. sine_matrix, coulomb_matrix)

    Returns
    -------

    representation:
        Np.ndarray of structure representation
    """
    if size is None:
        size = len(structure)

    if featurizer == "sine_matrix":
        sm = SineMatrix(n_atoms_max=size, permutation=permutation, **kwargs)
        rep = sm.create(structure, n_jobs=n_jobs).reshape(-1,)

    elif featurizer == "coulomb_matrix":
        cm = CoulombMatrix(n_atoms_max=size, permutation=permutation, **kwargs)
        rep = cm.create(structure, n_jobs=n_jobs).reshape(-1,)

    elif featurizer == "bob":
        qml_struct = ase_atoms_to_qml_compound(structure)
        qml_struct.generate_bob(size=size, **kwargs)
        return qml_struct.representation

    else:
        raise NotImplementedError("selected featurizer not implemented")

    return rep
