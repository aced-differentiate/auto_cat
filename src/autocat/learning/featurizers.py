from dscribe.descriptors import SineMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import ChemicalSRO
from matminer.featurizers.site import OPSiteFingerprint
from matminer.featurizers.site import CrystalNNFingerprint

import numpy as np
from prettytable import PrettyTable

from typing import List, Dict

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

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
        max_size: int = None,
        preset: str = None,
        kwargs: Dict = None,
    ):

        self._featurizer_class = None
        self.featurizer_class = featurizer_class

        self._preset = None
        self.preset = preset

        self._kwargs = None
        self.kwargs = kwargs

        self._max_size = 100
        self.max_size = max_size

        self._species_list = ["Pt", "Pd", "Cu", "Fe", "Ni", "H", "O", "C", "N"]
        self.species_list = species_list

        # overrides max_size and species_list if given
        self._design_space_structures = None
        self.design_space_structures = design_space_structures

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Featurizer"]
        class_name = (
            self.featurizer_class.__module__ + "." + self.featurizer_class.__name__
        )
        pt.add_row(["class", class_name])
        pt.add_row(["kwargs", self.kwargs])
        pt.add_row(["species list", self.species_list])
        pt.add_row(["maximum structure size", self.max_size])
        pt.add_row(["preset", self.preset])
        pt.add_row(
            [
                "design space structures provided?",
                self.design_space_structures is not None,
            ]
        )
        return str(pt)

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
            self._preset = None
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
            self._kwargs = kwargs.copy()

    @property
    def design_space_structures(self):
        return self._design_space_structures

    @design_space_structures.setter
    def design_space_structures(self, design_space_structures: List[Atoms]):
        if design_space_structures is not None:
            self._design_space_structures = [
                struct.copy() for struct in design_space_structures
            ]
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
            self._species_list = species_list.copy()

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

    def featurize_single(self, structure: Atoms):
        """
        Featurize a single structure. Returns a single vector

        Parameters
        ----------

        structure:
            ase.Atoms object of structure to be featurized

        Returns
        -------

        representation:
            Numpy array of feature vector (not flattened)
        """
        feat_class = self.featurizer_class
        if feat_class in SUPPORTED_DSCRIBE_CLASSES:
            if feat_class in [SOAP, ACSF]:
                adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
                return self.featurization_object.create(
                    structure, positions=adsorbate_indices,
                )
            elif feat_class in [SineMatrix, CoulombMatrix]:
                return self.featurization_object.create(structure).reshape(-1,)
        elif feat_class in SUPPORTED_MATMINER_CLASSES:
            conv = AseAtomsAdaptor()
            pym_struct = conv.get_structure(structure)
            if feat_class == ElementProperty:
                return np.array(
                    self.featurization_object.featurize(pym_struct.composition)
                )
            elif feat_class in [CrystalNNFingerprint, OPSiteFingerprint]:
                representation = np.array([])
                adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
                for idx in adsorbate_indices:
                    feat = self.featurization_object.featurize(pym_struct, idx)
                    representation = np.concatenate((representation, feat))
                return representation
            elif feat_class == ChemicalSRO:
                species_list = self.species_list
                adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
                formatted_list = [[pym_struct, idx] for idx in adsorbate_indices]
                featurization_object = self.featurization_object
                featurization_object.fit(formatted_list)
                # concatenate representation for each adsorbate atom
                representation = np.array([])
                # TODO: order species_list so that this is no longer needed
                for idx in adsorbate_indices:
                    raw_feat = featurization_object.featurize(pym_struct, idx)
                    # csro only generates for species observed in fit
                    # as well as includes, so to be generalizable
                    # we use full species list of the design space and place values
                    # in the appropriate species location relative to this list
                    labels = featurization_object.feature_labels()
                    feat = np.zeros(len(species_list))
                    for i, label in enumerate(labels):
                        # finds where corresponding species is in full species list
                        lbl_idx = np.where(
                            np.array(species_list) == label.split("_")[1]
                        )
                        feat[lbl_idx] = raw_feat[i]
                    representation = np.concatenate((representation, feat))
                return representation

    def featurize_multiple(self, structures: List[Atoms]):
        """
        Featurize multiple structures. Returns a matrix where each
        row is the flattened feature vector of each system

        Parameters
        ----------

        structures:
            List of ase.Atoms structures to be featurized

        Returns
        -------

        X:
            Numpy array of shape (number of structures, number of features)
        """
        first_vec = self.featurize_single(structures[0]).flatten()
        num_features = len(first_vec)
        # if adsorbate featurization, assumes only 1 adsorbate in design space
        # (otherwise would require padding)
        X = np.zeros((len(structures), num_features))
        X[0, :] = first_vec.copy()
        for i in range(1, len(structures)):
            X[i, :] = self.featurize_single(structures[i]).flatten()
        return X
