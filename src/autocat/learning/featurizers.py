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
from typing import Union

from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

# from autocat.io.qml import ase_atoms_to_qml_compound


def get_X(
    structures: List[Union[Atoms, str]],
    adsorbate_indices_dictionary: Dict[str, int],
    maximum_structure_size: int = None,
    structure_featurizer: str = "sine_matrix",
    maximum_adsorbate_size: int = None,
    adsorbate_featurizer: str = "soap",
    species_list: List[str] = None,
    structure_featurization_kwargs: Dict[str, float] = None,
    adsorbate_featurization_kwargs: Dict[str, float] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
):
    """
    Generate representation matrix X from list of ase.Atoms objects

    Parameters
    ----------

    structures:
        List of ase.Atoms objects or structure filename strings to be used to construct X matrix

    adsorbate_indices_dictionary:
        Dictionary mapping structures to desired adsorbate_indices
        (N.B. if structure is given as an ase.Atoms object,
        the key for this dictionary should be
        ase.Atoms.get_chemical_formula() + "_" + str(index in list))

    maximum_structure_size:
        Size of the largest structure to be supported by the representation.
        Default: number of atoms in largest structure within `structures`

    structure_featurizer:
        String giving featurizer to be used for full structure which will be
        fed into `full_structure_featurization`

    maximum_adsorbate_size:
        Integer giving the maximum adsorbate size to be encountered
        (ie. this determines if zero-padding should be applied and how much).
        If the provided value is less than the adsorbate size given by
        `adsorbate_indices`, representation will remain size of the adsorbate.
        Default: size of adsorbate provided

    adsorbate_featurizer:
        String giving featurizer to be used for full structure which will be
        fed into `adsorbate_structure_featurization`

    species_list:
        List of species that could be encountered for featurization.
        Default: Parses over all `structures` and collects all encountered species

    structure_featurization_kwargs:
        kwargs to be fed into `full_structure_featurization`

    adsorbate_featurization_kwargs:
        kwargs to be fed into `adsorbate_featurization`

    write_to_disk:
        Boolean specifying whether X should be written to disk as a json.
        Defaults to False.

    write_location:
        String with the location where X should be written to disk.

    Returns
    -------

    X_array:
        np.ndarray of X representation
        Shape is (# of structures, maximum_structure_size + maximum_adsorbate_size * # of adsorbates)
    """
    if maximum_structure_size is None:
        maximum_structure_size = max([len(s) for s in structures])

    if maximum_adsorbate_size is None:
        maximum_adsorbate_size = max(
            [len(adsorbate_indices_dictionary[a]) for a in adsorbate_indices_dictionary]
        )

    if adsorbate_featurization_kwargs is None:
        adsorbate_featurization_kwargs = {}

    if structure_featurization_kwargs is None:
        structure_featurization_kwargs = {}

    if species_list is None:
        species_list = []
        for s in structures:
            found_species = np.unique(s.get_chemical_symbols()).tolist()
            new_species = [spec for spec in found_species if spec not in species_list]
            species_list.extend(new_species)

    if adsorbate_featurizer is not None:
        num_of_adsorbate_features = _get_number_of_features(
            featurizer=adsorbate_featurizer,
            species=species_list,
            **adsorbate_featurization_kwargs,
        )

    if (structure_featurizer is not None) and (adsorbate_featurizer is not None):
        X = np.zeros(
            (
                len(structures),
                maximum_structure_size ** 2
                + maximum_adsorbate_size * num_of_adsorbate_features,
            )
        )
    elif (structure_featurizer is None) and (adsorbate_featurizer is not None):
        X = np.zeros(
            (len(structures), maximum_adsorbate_size * num_of_adsorbate_features)
        )

    elif (structure_featurizer is not None) and (adsorbate_featurizer is None):
        X = np.zeros((len(structures), maximum_structure_size ** 2))
    else:
        msg = "Need to specify either a structure or adsorbate featurizer"
        raise ValueError(msg)

    for idx, structure in enumerate(structures):
        if isinstance(structure, Atoms):
            name = structure.get_chemical_formula() + "_" + str(idx)
            ase_struct = structure.copy()
        elif isinstance(structure, str):
            name = structure
            ase_struct = read(structure)
        else:
            raise TypeError(f"Each structure needs to be either a str or ase.Atoms")
        cat_feat = catalyst_featurization(
            ase_struct,
            adsorbate_indices=adsorbate_indices_dictionary[name],
            structure_featurizer=structure_featurizer,
            adsorbate_featurizer=adsorbate_featurizer,
            maximum_structure_size=maximum_structure_size,
            maximum_adsorbate_size=maximum_adsorbate_size,
            species_list=species_list,
            structure_featurization_kwargs=structure_featurization_kwargs,
            adsorbate_featurization_kwargs=adsorbate_featurization_kwargs,
        )
        X[idx] = cat_feat

    if write_to_disk:
        if not os.path.isdir(write_location):
            os.makedirs(write_location)
        write_path = os.path.join(write_location, "X.json")
        with open(write_path, "w") as f:
            json.dump(X.tolist(), f)
        print(f"X written to {write_path}")

    return X


def catalyst_featurization(
    structure: Atoms,
    adsorbate_indices: List[int],
    structure_featurizer: str = "sine_matrix",
    adsorbate_featurizer: str = "soap",
    maximum_structure_size: int = None,
    maximum_adsorbate_size: int = None,
    species_list: List[str] = None,
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
    if structure_featurization_kwargs is None:
        structure_featurization_kwargs = {}

    if adsorbate_featurization_kwargs is None:
        adsorbate_featurization_kwargs = {}

    if structure_featurizer is not None:
        struct_feat = full_structure_featurization(
            structure,
            featurizer=structure_featurizer,
            maximum_structure_size=maximum_structure_size,
            **structure_featurization_kwargs,
        )
    else:
        struct_feat = np.array([])

    if adsorbate_featurizer is not None:
        ads_feat = adsorbate_featurization(
            structure,
            featurizer=adsorbate_featurizer,
            adsorbate_indices=adsorbate_indices,
            species_list=species_list,
            maximum_adsorbate_size=maximum_adsorbate_size,
            **adsorbate_featurization_kwargs,
        )
    else:
        ads_feat = np.array([])
    cat_feat = np.concatenate((struct_feat, ads_feat))
    return cat_feat


def adsorbate_featurization(
    structure: Atoms,
    adsorbate_indices: List[int],
    species_list: List[str] = None,
    featurizer: str = "soap",
    rcut: float = 6.0,
    maximum_adsorbate_size: int = None,
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

    maximum_adsorbate_size:
        Integer giving the maximum adsorbate size to be encountered
        (ie. this determines if zero-padding should be applied and how much).
        If the provided value is less than the adsorbate size given by
        `adsorbate_indices`, representation will remain size of the adsorbate.
        Default: size of adsorbate provided

    Returns
    -------

    representation:
        Np.ndarray of adsorbate representation

    """
    # Checks if species list given
    if species_list is None:
        species_list = np.unique(structure.get_chemical_symbols()).tolist()

    # Checks if max adsorbate size specified
    if maximum_adsorbate_size is None:
        maximum_adsorbate_size = len(adsorbate_indices)

    # Selects appropriate featurizer
    if featurizer == "soap":
        soap = SOAP(rcut=rcut, species=species_list, **kwargs)
        representation = soap.create(structure, positions=adsorbate_indices).reshape(
            -1,
        )
        num_of_features = soap.get_number_of_features()

    elif featurizer == "acsf":
        acsf = ACSF(rcut=rcut, species=species_list, **kwargs)
        representation = acsf.create(structure, positions=adsorbate_indices).reshape(
            -1,
        )
        num_of_features = acsf.get_number_of_features()

    else:
        raise NotImplementedError("selected featurizer not implemented")

    # Checks if padding needs to be applied
    if len(representation) < maximum_adsorbate_size * num_of_features:
        diff = maximum_adsorbate_size * num_of_features - len(representation)
        representation = np.pad(representation, (0, diff))

    return representation


def full_structure_featurization(
    structure: Atoms,
    maximum_structure_size: int = None,
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

    maximum_structure_size:
        Size of the largest structure to be supported by the representation.
        Default: number of atoms in `structures`

    featurizer:
        String indicating featurizer to be used.

        Options:
        - sine_matrix (default)
        - coulomb_matrix (N.B.: does not support periodicity)

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
    if maximum_structure_size is None:
        maximum_structure_size = len(structure)

    if featurizer == "sine_matrix":
        sm = SineMatrix(
            n_atoms_max=maximum_structure_size, permutation=permutation, **kwargs
        )
        rep = sm.create(structure, n_jobs=n_jobs).reshape(-1,)

    elif featurizer == "coulomb_matrix":
        cm = CoulombMatrix(
            n_atoms_max=maximum_structure_size, permutation=permutation, **kwargs
        )
        rep = cm.create(structure, n_jobs=n_jobs).reshape(-1,)

    else:
        raise NotImplementedError("selected featurizer not implemented")

    return rep


def _get_number_of_features(featurizer, **kwargs):
    """
    Wrapper of `get_number_of_features` method for `dscribe`
    featurizers
    """
    supported_featurizers = {
        "sine_matrix": SineMatrix,
        "coulomb_matrix": CoulombMatrix,
        "soap": SOAP,
        "acsf": ACSF,
    }

    if featurizer not in supported_featurizers:
        raise NotImplementedError(
            "selected featurizer does not currently support this feature"
        )

    feat = supported_featurizers[featurizer](**kwargs)
    return feat.get_number_of_features()
