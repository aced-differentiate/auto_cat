import copy
from typing import List, Dict

import numpy as np
from prettytable import PrettyTable

from ase import Atoms
from dscribe.descriptors import SineMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import ChemicalSRO
from matminer.featurizers.site import OPSiteFingerprint
from matminer.featurizers.site import CrystalNNFingerprint
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element


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
        featurizer_class=None,  # black
        design_space_structures: List[Atoms] = None,
        species_list: List[str] = None,
        max_size: int = None,
        preset: str = None,
        kwargs: Dict = None,
    ):

        self._featurizer_class = SineMatrix
        self.featurizer_class = featurizer_class

        self._preset = None
        self.preset = preset

        self._kwargs = None
        self.kwargs = kwargs

        self._max_size = 100
        self.max_size = max_size

        self._species_list = ["Fe", "Ni", "Pt", "Pd", "Cu", "C", "N", "O", "H"]
        self.species_list = species_list

        # overrides max_size and species_list if given
        self._design_space_structures = None
        self.design_space_structures = design_space_structures

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Featurizer):
            for attr in [
                "featurizer_class",
                "species_list",
                "max_size",
                "preset",
                "kwargs",
            ]:
                if getattr(self, attr) != getattr(other, attr):
                    return False
            return True
        return False

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
        pt.max_width = 70
        return str(pt)

    def copy(self):
        """
        Returns a copy of the featurizer
        """
        ds_structs_copy = (
            [struct.copy() for struct in self.design_space_structures]
            if self.design_space_structures
            else None
        )
        feat = self.__class__(
            featurizer_class=self.featurizer_class,
            design_space_structures=ds_structs_copy,
            species_list=self.species_list.copy(),
            max_size=self.max_size,
            kwargs=copy.deepcopy(self.kwargs) if self.kwargs else None,
        )
        return feat

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
            _species_list = []
            for s in ds_structs:
                # get all unique species
                found_species = np.unique(s.get_chemical_symbols()).tolist()
                new_species = [
                    spec for spec in found_species if spec not in _species_list
                ]
                _species_list.extend(new_species)
            # sort species list
            sorted_species_list = sorted(
                _species_list, key=lambda el: Element(el).mendeleev_no
            )

            self._max_size = max([len(s) for s in ds_structs])
            self._species_list = sorted_species_list

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
            _species_list = species_list.copy()
            # sort species list by mendeleev number
            sorted_species_list = sorted(
                _species_list, key=lambda el: Element(el).mendeleev_no
            )
            self._species_list = sorted_species_list

    # TODO: "get_featurization_object" -> "get_featurizer"
    @property
    def featurization_object(self):
        return self._get_featurization_object()

    def _get_featurization_object(self):
        # instantiate featurizer object
        if hasattr(self.featurizer_class, "from_preset") and self.preset is not None:
            return self.featurizer_class.from_preset(self.preset)
        if self.featurizer_class in [SineMatrix, CoulombMatrix]:
            return self.featurizer_class(
                n_atoms_max=self.max_size, permutation="none", **self.kwargs or {},
            )
        if self.featurizer_class in [SOAP, ACSF]:
            return self.featurizer_class(species=self.species_list, **self.kwargs or {})
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
        featurization_object = self.featurization_object
        # dscribe classes
        if feat_class in [SOAP, ACSF]:
            adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
            return featurization_object.create(structure, positions=adsorbate_indices,)
        if feat_class in [SineMatrix, CoulombMatrix]:
            return featurization_object.create(structure).reshape(-1,)

        # matminer classes
        pym_struct = AseAtomsAdaptor().get_structure(structure)
        if feat_class == ElementProperty:
            return np.array(featurization_object.featurize(pym_struct.composition))
        representation = np.array([])
        if feat_class in [CrystalNNFingerprint, OPSiteFingerprint]:
            adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
            for idx in adsorbate_indices:
                feat = featurization_object.featurize(pym_struct, idx)
                representation = np.concatenate((representation, feat))
            return representation
        if feat_class == ChemicalSRO:
            adsorbate_indices = np.where(structure.get_tags() <= 0)[0].tolist()
            formatted_list = [[pym_struct, idx] for idx in adsorbate_indices]
            featurization_object.fit(formatted_list)
            for idx in adsorbate_indices:
                feat = featurization_object.featurize(pym_struct, idx)
                representation = np.concatenate((representation, feat))
            return representation
        return None

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
