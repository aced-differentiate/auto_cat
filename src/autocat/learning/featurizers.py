from dscribe.descriptors import SineMatrix
from dscribe.descriptors import EwaldSumMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import ChemicalSRO
from matminer.featurizers.site import OPSiteFingerprint
from matminer.featurizers.site import CrystalNNFingerprint

import os
import numpy as np
import json

from typing import List, Dict
from typing import Union

from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN

SUPPORTED_MATMINER_CLASSES = [
    ElementProperty,
    ChemicalSRO,
    OPSiteFingerprint,
    CrystalNNFingerprint,
]

SUPPORTED_DSCRIBE_CLASSES = [SineMatrix, CoulombMatrix, ACSF, SOAP]


class FeaturizerError(Exception):
    pass


class Featurizer:
    def __init__(
        self,
        featurizer_class,
        design_space_structures: List[Atoms] = None,
        species_list: List[str] = None,
        max_size: int = 100,
        preset: str = None,
        kwargs: Dict = None,
    ):

        self._featurizer_class = None
        self.featurizer_class = featurizer_class

        self._preset = None
        self.preset = preset

        self._kwargs = None
        self.kwargs = kwargs

        self._max_size = None
        self.max_size = max_size

        self._species_list = ["Pt", "Pd", "Cu", "Fe", "Ni", "H", "O", "C", "N"]
        self.species_list = species_list

        # overrides max_size and species_list if given
        self._design_space_structures = None
        self.design_space_structures = design_space_structures

        self._featurizer_object = self._get_featurization_object()

    @property
    def featurizer_class(self):
        return self._featurizer_class

    @featurizer_class.setter
    def featurizer_class(self, featurizer_class):
        if (
            featurizer_class in SUPPORTED_MATMINER_CLASSES
            or featurizer_class in SUPPORTED_DSCRIBE_CLASSES
        ):
            self._featurizer_class = featurizer_class
            self._kwargs = None
        else:
            msg = f"Featurization class {featurizer_class} is not currently supported."
            raise FeaturizerError(msg)

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, preset):
        if self.featurizer_class in [CrystalNNFingerprint, ElementProperty]:
            self._preset = preset
        elif preset is None:
            self._preset = preset
        else:
            msg = f"Presets are not supported for {self.featurizer_class.__module__}"
            raise FeaturizerError(msg)

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        if kwargs is not None:
            self._kwargs = kwargs

    @property
    def design_space_structures(self):
        return self._design_space_structures

    @design_space_structures.setter
    def design_space_structures(self, design_space_structures: List[Atoms]):
        if design_space_structures is not None:
            self._design_space_structures = design_space_structures
            # analyze new design space
            ds_structs = design_space_structures
            species_list = []
            for s in ds_structs:
                # get all unique species
                found_species = np.unique(s.get_chemical_symbols()).tolist()
                new_species = [
                    spec for spec in found_species if spec not in species_list
                ]
                species_list.extend(new_species)

            self._max_size = max([len(s) for s in ds_structs])
            self._species_list = species_list

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, max_size):
        if max_size is not None:
            self._max_size = max_size

    @property
    def species_list(self):
        return self._species_list

    @species_list.setter
    def species_list(self, species_list: List[str]):
        if species_list is not None:
            self._species_list = species_list

    @property
    def featurization_object(self):
        return self._get_featurization_object()

    def _get_featurization_object(self):
        # instantiate featurizer object
        if self.preset is not None:
            try:
                return self.featurizer_class.from_preset(self.preset)
            except:
                msg = f"{self.featurizer_class} cannot be initialized from the preset {self.preset}"
                raise FeaturizerError(msg)
        else:
            if self.featurizer_class in [SineMatrix, CoulombMatrix]:
                return self.featurizer_class(
                    n_atoms_max=self.max_size, permutation="none", **self.kwargs or {},
                )
            elif self.featurizer_class in [SOAP, ACSF]:
                return self.featurizer_class(
                    species=self.species_list, **self.kwargs or {}
                )
            return self.featurizer_class(**self.kwargs or {})

    def featurize_single(self, structure: Atoms, **kwargs):
        feat_class = self.featurizer_class
        if feat_class in SUPPORTED_DSCRIBE_CLASSES:
            if feat_class in [SOAP, ACSF]:
                adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
                return self.featurizer_object.create(
                    structure, positions=adsorbate_indices, **kwargs
                )
            elif feat_class in [SineMatrix, CoulombMatrix]:
                return self.featurizer_object.create(structure, **kwargs)
        elif feat_class in SUPPORTED_MATMINER_CLASSES:
            conv = AseAtomsAdaptor()
            pym_struct = conv.get_structure(structure)
            if feat_class == ElementProperty:
                return np.array(
                    self.featurizer_object.featurize(pym_struct.composition)
                )
            elif feat_class in [CrystalNNFingerprint, OPSiteFingerprint]:
                representation = np.array([])
                adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
                for idx in adsorbate_indices:
                    feat = self.featurizer_object.featurize(pym_struct, idx)
                    representation = np.concatenate((representation, feat))
                return representation
            elif feat_class == ChemicalSRO:
                species_list = self.species_list
                adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
                formatted_list = [[pym_struct, idx] for idx in adsorbate_indices]
                self.featurizer_object.fit(formatted_list)
                # concatenate representation for each adsorbate atom
                representation = np.array([])
                for idx in adsorbate_indices:
                    raw_feat = self.featurizer_object.featurize(pym_struct, idx)
                    # csro only generates for species observed in fit
                    # as well as includes, so to be generalizable
                    # we use full species list of the design space and place values
                    # in the appropriate species location relative to this list
                    labels = self.featurizer_object.feature_labels()
                    feat = np.zeros(len(species_list))
                    for i, label in enumerate(labels):
                        # finds where corresponding species is in full species list
                        lbl_idx = np.where(
                            np.array(species_list) == label.split("_")[1]
                        )
                        feat[lbl_idx] = raw_feat[i]
                    representation = np.concatenate((representation, feat))
                return representation

    def featurize_multiple(self, structures: List[Atoms], **kwargs):
        first_vec = self.featurize_single(structures[0], **kwargs)
        num_features = len(first_vec)
        # if adsorbate featurization, assumes only 1 adsorbate in design space
        # (otherwise would require padding)
        X = np.zeros((len(structures), num_features))
        X[0, :] = first_vec.copy()
        for i in range(1, len(structures)):
            X[i, :] = self.featurize_single(structures[i], **kwargs)
        return X


def get_X(
    structures: List[Union[Atoms, str]],
    maximum_structure_size: int = None,
    structure_featurizer: str = "sine_matrix",
    elementalproperty_preset: str = "magpie",
    maximum_adsorbate_size: int = None,
    adsorbate_featurizer: str = "soap",
    species_list: List[str] = None,
    refine_structures: bool = True,
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

    refine_structures:
        Bool indicating whether the structures should be refined to include
        only the adsorbate and surface layer. Requires tags for all structures
        to have adsorbate atoms and surface atoms as 0 and 1, respectively

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
        if refine_structures:
            ref_structures = [
                structure[np.where(structure.get_tags() < 2)[0].tolist()]
                for structure in structures
            ]
            maximum_structure_size = max([len(ref) for ref in ref_structures])
        else:
            maximum_structure_size = max([len(s) for s in structures])

    if maximum_adsorbate_size is None:
        adsorbate_sizes = []
        for struct in structures:
            adsorbate_sizes.append(len(np.where(struct.get_tags() <= 0)[0].tolist()))
        maximum_adsorbate_size = max(adsorbate_sizes)

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
        raise FeaturizerError(msg)

    for idx, structure in enumerate(structures):
        if isinstance(structure, Atoms):
            ase_struct = structure.copy()
        elif isinstance(structure, str):
            ase_struct = read(structure)
        else:
            msg = f"Each structure needs to be either a str or ase.Atoms. Got {type(structure)}"
            raise FeaturizerError(msg)
        cat_feat = catalyst_featurization(
            ase_struct,
            structure_featurizer=structure_featurizer,
            adsorbate_featurizer=adsorbate_featurizer,
            elementalproperty_preset=elementalproperty_preset,
            maximum_structure_size=maximum_structure_size,
            maximum_adsorbate_size=maximum_adsorbate_size,
            species_list=species_list,
            refine_structure=refine_structures,
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
    structure_featurizer: str = "sine_matrix",
    elementalproperty_preset: str = "magpie",
    adsorbate_featurizer: str = "soap",
    maximum_structure_size: int = None,
    maximum_adsorbate_size: int = None,
    species_list: List[str] = None,
    structure_featurization_kwargs: Dict[str, float] = None,
    adsorbate_featurization_kwargs: Dict[str, float] = None,
    refine_structure: bool = True,
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

    refine_structure:
        Bool indicating whether the structure should be refined to include
        only the adsorbate and surface layer. Requires tags for the structure
        to have adsorbate atoms and surface atoms as 0 and 1, respectivel

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
            refine_structure=refine_structure,
            **structure_featurization_kwargs,
        )
    else:
        struct_feat = np.array([])

    if adsorbate_featurizer is not None:
        ads_feat = adsorbate_featurization(
            structure,
            featurizer=adsorbate_featurizer,
            species_list=species_list,
            maximum_adsorbate_size=maximum_adsorbate_size,
            refine_structure=refine_structure,
            **adsorbate_featurization_kwargs,
        )
    else:
        ads_feat = np.array([])
    cat_feat = np.concatenate((struct_feat, ads_feat))
    return cat_feat


def adsorbate_featurization(
    structure: Atoms,
    species_list: List[str] = None,
    featurizer: str = "soap",
    rcut: float = 6.0,
    maximum_adsorbate_size: int = None,
    refine_structure: bool = True,
    **kwargs,
):
    """
    Featurizes adsorbate on a support via `dscribe`
    and `matminer.featurizers.site` functions.

    Parameters
    ----------

    structure:
        Atoms object of full structure (adsorbate + slab) to be featurized

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
        op_sitefingerprint: order parameter site fingerprint

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

    refine_structure:
        Bool indicating whether the structure should be refined to include
        only the adsorbate and surface layer. Requires tags for the structure
        to have adsorbate atoms and surface atoms as 0 and 1, respectivel

    Returns
    -------

    representation:
        Np.ndarray of adsorbate representation

    """
    adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()

    if refine_structure:
        new_indices = np.where(structure.get_tags() < 2)[0].tolist()
        # update the adsorbate indices for refined structure
        new_adsorbate_indices = []
        for ads_idx in adsorbate_indices:
            # wraps index
            if ads_idx < 0:
                ads_idx = len(structure) + ads_idx
            new_adsorbate_indices.append(new_indices.index(ads_idx))
        adsorbate_indices = new_adsorbate_indices
        # refine the structure
        structure = structure[new_indices]

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

    elif featurizer == "op_sitefingerprint":
        opsf = OPSiteFingerprint(**kwargs)
        conv = AseAtomsAdaptor()
        pym_struct = conv.get_structure(structure)
        representation = np.array([])
        for idx in adsorbate_indices:
            feat = opsf.featurize(pym_struct, idx)
            representation = np.concatenate((representation, feat))
        num_of_features = len(opsf.feature_labels())

    elif featurizer == "crystalnn_sitefingerprint":
        cnn = CrystalNNFingerprint.from_preset("cn")
        conv = AseAtomsAdaptor()
        pym_struct = conv.get_structure(structure)
        representation = np.array([])
        for idx in adsorbate_indices:
            feat = cnn.featurize(pym_struct, idx)
            representation = np.concatenate((representation, feat))
        num_of_features = len(cnn.feature_labels())

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
    refine_structure: bool = True,
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
        - elemental_property

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

    refine_structure:
        Bool indicating whether the structure should be refined to include
        only the adsorbate and surface layer. Requires tags for the structure
        to have adsorbate atoms and surface atoms as 0 and 1, respectively

    Returns
    -------

    representation:
        Np.ndarray of structure representation
    """

    if refine_structure:
        structure = structure[np.where(structure.get_tags() < 2)[0].tolist()]

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
        "op_sitefingerprint": OPSiteFingerprint,
        "crystalnn_sitefingerprint": CrystalNNFingerprint,
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
        elif featurizer == "op_sitefingerprint":
            f = supported_matminer_featurizers[featurizer](**kwargs)
            return len(f.feature_labels())
        elif featurizer == "crystalnn_sitefingerprint":
            f = supported_matminer_featurizers[featurizer].from_preset("cn")
            return len(f.feature_labels())

    else:
        raise NotImplementedError(
            "selected featurizer does not currently support this feature"
        )
