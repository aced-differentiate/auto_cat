from dscribe.descriptors import SineMatrix
from dscribe.descriptors import EwaldSumMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import ChemicalSRO

import tempfile
import os
import numpy as np
import json

from typing import List, Dict
from typing import Union

from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN

# from autocat.io.qml import ase_atoms_to_qml_compound


class AutoCatFeaturizationError(Exception):
    pass


def get_X(
    structures: List[Union[Atoms, str]],
    adsorbate_indices_dictionary: Dict[str, int] = None,
    maximum_structure_size: int = None,
    structure_featurizer: str = "sine_matrix",
    elementalproperty_preset: str = "magpie",
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
        Dictionary mapping structures to desired adsorbate_indices.
        Only necessary if adsorbate featurization is specified
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
        # check that adsorbate indices specified
        if adsorbate_indices_dictionary is None:
            msg = "For adsorbate featurization, adsorbate indices must be specified"
            raise AutoCatFeaturizationError(msg)
        # find number of adsorbate features
        if adsorbate_featurizer in ["soap", "acsf"]:
            num_of_adsorbate_features = _get_number_of_features(
                featurizer=adsorbate_featurizer,
                species=species_list,
                **adsorbate_featurization_kwargs,
            )
        else:
            # chemical_sro
            num_of_adsorbate_features = _get_number_of_features(
                featurizer=adsorbate_featurizer,
                species=species_list,
                **adsorbate_featurization_kwargs,
            )

    if structure_featurizer is not None:
        if structure_featurizer in ["sine_matrix", "coulomb_matrix"]:
            num_structure_features = maximum_structure_size ** 2
        else:
            # elemental property featurizer
            num_structure_features = _get_number_of_features(
                structure_featurizer,
                elementalproperty_preset=elementalproperty_preset,
                **structure_featurization_kwargs,
            )

    if (structure_featurizer is not None) and (adsorbate_featurizer is not None):
        X = np.zeros(
            (
                len(structures),
                num_structure_features
                + maximum_adsorbate_size * num_of_adsorbate_features,
            )
        )
    elif (structure_featurizer is None) and (adsorbate_featurizer is not None):
        X = np.zeros(
            (len(structures), maximum_adsorbate_size * num_of_adsorbate_features)
        )

    elif (structure_featurizer is not None) and (adsorbate_featurizer is None):
        X = np.zeros((len(structures), num_structure_features))
    else:
        msg = "Need to specify either a structure or adsorbate featurizer"
        raise AutoCatFeaturizationError(msg)

    for idx, structure in enumerate(structures):
        if isinstance(structure, Atoms):
            name = structure.get_chemical_formula() + "_" + str(idx)
            ase_struct = structure.copy()
        elif isinstance(structure, str):
            name = structure
            ase_struct = read(structure)
        else:
            msg = f"Each structure needs to be either a str or ase.Atoms. Got {type(structure)}"
            raise AutoCatFeaturizationError(msg)
        # if no adsorbate featurizer, makes placeholder dict for each loop
        if adsorbate_featurizer is None:
            adsorbate_indices_dictionary = {name: None}
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
    adsorbate_indices: List[int] = None,
    structure_featurizer: str = "sine_matrix",
    elementalproperty_preset: str = "magpie",
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
        featurized.
        Only necessary if adsorbate featurization is specified

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
            elementalproperty_preset=elementalproperty_preset,
            **structure_featurization_kwargs,
        )
    else:
        struct_feat = np.array([])

    if adsorbate_featurizer is not None:
        if adsorbate_indices is None:
            msg = "For adsorbate featurization, adsorbate indices must be specified"
            raise AutoCatFeaturizationError(msg)
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
        which is fed into `dscribe.descriptors.{ACSF,SOAP}` or
        `ChemicalSRO.includes`
        (ie. any species expected to be encountered)
        Default: species present in `structure`

    featurizer:
        String indicating featurizer to be used.

        Options:
        acsf: atom centered symmetry functions
        soap (default): smooth overlap of atomic positions
        chemical_sro: chemical short range ordering

    rcut:
        Float giving cutoff radius to be used when generating
        representation. For chemical_sro, this is the cutoff used
        for determining the nearest neighbors
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
    # Ensures that adsorbate indices specified are sorted
    adsorbate_indices.sort()

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

    elif featurizer == "chemical_sro":
        # generate nn calculator
        vnn = VoronoiNN(cutoff=rcut, allow_pathological=True)
        csro = ChemicalSRO(vnn, includes=species_list)
        # convert ase structure to pymatgen
        conv = AseAtomsAdaptor()
        pym_struct = conv.get_structure(structure)
        # format list for csro fitting (to get species)
        formatted_list = [[pym_struct, idx] for idx in adsorbate_indices]
        csro.fit(formatted_list)
        # concatenate representation for each adsorbate atom
        representation = np.array([])
        for idx in adsorbate_indices:
            raw_feat = csro.featurize(pym_struct, idx)
            # csro only generates for species observed in fit
            # as well as includes, so to be generalizable
            # we use full species list and place values
            # in the appropriate species location
            labels = csro.feature_labels()
            feat = np.zeros(len(species_list))
            for i, label in enumerate(labels):
                # finds where corresponding species is in full species list
                lbl_idx = np.where(np.array(species_list) == label.split("_")[1])
                feat[lbl_idx] = raw_feat[i]
            representation = np.concatenate((representation, feat))
        # number of features is number of species specified
        num_of_features = len(species_list)

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
    elementalproperty_preset: str = "magpie",
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
        - element property

    permutation:
        String specifying how ordering is handled. This is fed into
        `dscribe` featurizers (ie. sine_matrix, ewald_sum_matrix, coulomb_matrix)
        Default: "none", maintains same ordering as input Atoms structure
        (N.B. this differs from the `dscribe` default)

    elementalproperty_preset:
        String giving the preset to be pulled from.
        Options:
        - magpie (default)
        - pymatgen
        - deml
        - matscholar_el
        - megnet_el
        See `matminer` documentation for more details

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

    elif featurizer == "elemental_property":
        ep = ElementProperty.from_preset(elementalproperty_preset)
        conv = AseAtomsAdaptor()
        pymat = conv.get_structure(structure)
        rep = np.array(ep.featurize(pymat.composition))

    else:
        raise NotImplementedError("selected featurizer not implemented")

    return rep


def _get_number_of_features(
    featurizer,
    elementalproperty_preset: str = "magpie",
    species: List[str] = None,
    **kwargs,
):
    """
    Helper function to get number of features.

    Wrapper of `get_number_of_features` method for `dscribe`
    featurizers

    If `matminer`'s elemental property, calculated based off of
    number of features X number of stats
    """
    supported_dscribe_featurizers = {
        "sine_matrix": SineMatrix,
        "coulomb_matrix": CoulombMatrix,
        "soap": SOAP,
        "acsf": ACSF,
    }

    supported_matminer_featurizers = {
        "elemental_property": ElementProperty.from_preset(elementalproperty_preset),
        "chemical_sro": None,
    }

    if featurizer in supported_dscribe_featurizers:
        feat = supported_dscribe_featurizers[featurizer](species=species, **kwargs)
        return feat.get_number_of_features()

    elif featurizer in supported_matminer_featurizers:
        if featurizer == "elemental_property":
            ep = supported_matminer_featurizers[featurizer]
            return len(ep.features) * len(ep.stats)
        elif featurizer == "chemical_sro":
            return len(species)

    else:
        raise NotImplementedError(
            "selected featurizer does not currently support this feature"
        )
