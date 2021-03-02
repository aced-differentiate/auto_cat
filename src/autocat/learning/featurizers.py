import qml

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import EwaldSumMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

import tempfile
import os
import numpy as np
import json

from typing import List, Dict

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


def catalyst_featurization(
    structure: Atoms,
    adsorbate_indices: List[int],
    structure_featurizer: str = "sine_matrix",
    adsorbate_featurizer: str = "acsf",
    structure_featurization_kwargs: Dict[str, float] = None,
    adsorbate_featurization_kwargs: Dict[str, float] = None,
):
    """
    Featurizes a system containing an adsorbate + substrate
    in terms of both a full structure and adsorbate
    featurization via concatenation.

    In other words,
    catalyst_featurization = full_structure_featurization + adsorbate_featurization

    Parameters
    ----------

    structure:
        Atoms object of full structure (adsorbate + slab) to be featurized

    adsorbate_indices:
        List of atomic indices specifying the adsorbate to be
        featurized

    structure_featurizer:
        String giving featurizer to be used for full structure which will be
        fed into `full_structure_featurization`

    adsorbate_featurizer:
        String giving featurizer to be used for full structure which will be
        fed into `adsorbate_structure_featurization`

    structure_featurization_kwargs:
        kwargs to be fed into `full_structure_featurization`

    adsorbate_featurization_kwargs:
        kwargs to be fed into `adsorbate_featurization`

    Returns
    -------

    cat_feat:
        Np.ndarray of featurized structure

    """

    struct_feat = full_structure_featurization(
        structure, **structure_featurization_kwargs, featurizer=structure_featurizer
    )
    ads_feat = adsorbate_featurization(
        structure,
        featurizer=adsorbate_featurizer,
        adsorbate_indices=adsorbate_indices,
        **adsorbate_featurization_kwargs,
    )
    cat_feat = np.concatenate((struct_feat, ads_feat))
    return cat_feat


def adsorbate_featurization(
    structure: Atoms,
    adsorbate_indices: List[int],
    species_list: List[str] = None,
    featurizer: str = "acsf",
    rcut: float = 6.0,
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

    species_list:
        List of chemical species that should be covered by representation
        which is fed into `dscribe.descriptors.{ACSF,SOAP}`
        (ie. any species expected to be encountered)
        Default: species present in `structure`

    featurizer:
        String indicating featurizer to be used.

        Options:
        acsf (default): atom centered symmetry functions
        soap: smooth overlap of atomic positions

    rcut:
        Float giving cutoff radius to be used when generating
        representation.
        Default: 6 angstroms

    Returns
    -------

    representation:
        Np.ndarray of adsorbate representation

    """
    if species_list is None:
        species_array = np.unique(structure.get_chemical_symbols())
        species_list = species_array.tolist()

    if featurizer == "acsf":
        acsf = ACSF(rcut=rcut, species=species_list, **kwargs)
        representation = acsf.create(structure, positions=adsorbate_indices).reshape(
            -1,
        )

    elif featurizer == "soap":
        soap = SOAP(rcut=rcut, species=species_list, **kwargs)
        representation = soap.create(structure, positions=adsorbate_indices).reshape(
            -1,
        )

    else:
        raise NotImplementedError("selected featurizer not implemented")

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
        - bob (bag of bonds)

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
