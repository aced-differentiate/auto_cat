import copy
import os
import json
import itertools
from typing import List
from typing import Dict
from typing import Union

import numpy as np
from joblib import Parallel, delayed
from prettytable import PrettyTable
from ase import Atoms
from ase.io.jsonio import encode as atoms_encoder
from ase.io.jsonio import decode as atoms_decoder
from scipy import stats
from olympus import Surface

from autocat.learning.predictors import Predictor
from autocat.data.hhi import HHI
from autocat.data.segregation_energies import SEGREGATION_ENERGIES


Array = List[float]


class DesignSpaceError(Exception):
    pass


class DesignSpace:
    def __init__(
        self,
        design_space_structures: List[Atoms] = None,
        design_space_labels: Array = None,
        feature_matrix: Array = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        design_space_labels:
            Labels corresponding to all structures within the design space.
            If label not yet known, set to np.nan

        design_space_structures:
            List of all structures within the design space

        feature_matrix:
            Feature matrix of full design space. Takes priority over
            `design_space_structures` if provided

        """
        if feature_matrix is None and design_space_structures is None:
            msg = "Either a feature matrix or list of design space structures\
                 must be provided"
            raise DesignSpaceError(msg)

        self._design_space_labels = design_space_labels.copy()

        self._feature_matrix = None
        if feature_matrix is not None:
            if len(feature_matrix) != design_space_labels.shape[0]:
                msg = f"Number of rows ({len(feature_matrix)})\
                     and labels ({design_space_labels.shape[0]}) must match"
                raise DesignSpaceError(msg)
            self._feature_matrix = feature_matrix

        if design_space_structures is not None:
            if len(design_space_structures) != design_space_labels.shape[0]:
                msg = f"Number of structures ({len(design_space_structures)})\
                     and labels ({design_space_labels.shape[0]}) must match"
                raise DesignSpaceError(msg)

            self._design_space_structures = [
                struct.copy() for struct in design_space_structures
            ]
        else:
            self._design_space_structures = None

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "DesignSpace"]
        pt.add_row(["total # of systems", len(self)])
        num_unknown = sum(np.isnan(self.design_space_labels))
        pt.add_row(["# of unlabelled systems", num_unknown])
        pt.add_row(
            [
                "unique species present",
                self.species_list if self.species_list else "unknown",
            ]
        )
        max_label = max(self.design_space_labels)
        pt.add_row(["maximum label", max_label])
        min_label = min(self.design_space_labels)
        pt.add_row(["minimum label", min_label])
        pt.max_width = 70
        return str(pt)

    def __len__(self):
        return (
            len(self.feature_matrix)
            if self.feature_matrix is not None
            else len(self.design_space_structures)
        )

    # TODO: non-dunder method for deleting systems
    def __delitem__(self, i):
        """
        Deletes systems from the design space. If mask provided, deletes wherever True
        """
        if isinstance(i, list):
            i = np.array(i)
        elif isinstance(i, int):
            i = [i]
        mask = np.ones(len(self), dtype=bool)
        mask[i] = 0
        self._design_space_labels = self.design_space_labels[mask]
        if self.feature_matrix is not None:
            self._feature_matrix = self.feature_matrix[mask]
        if self.design_space_structures:
            structs = self.design_space_structures
            masked_structs = [structs[j] for j in range(len(self)) if mask[j]]
            self._design_space_structures = masked_structs

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DesignSpace):
            # check that they are the same length
            if len(self) == len(other):
                # check all their structures are equal
                self_structs = self.design_space_structures
                o_structs = other.design_space_structures
                if not self_structs == o_structs:
                    return False

                # check feature matrices are equal
                self_feature_matrix = self.feature_matrix
                o_feature_matrix = other.feature_matrix
                if not np.array_equal(self_feature_matrix, o_feature_matrix):
                    return False

                # check their labels are equal
                self_labels = self.design_space_labels
                o_labels = other.design_space_labels
                return np.array_equal(self_labels, o_labels, equal_nan=True)
        return False

    def copy(self):
        """
        Returns a copy of the design space
        """
        acds = self.__class__(
            design_space_structures=self.design_space_structures,
            design_space_labels=self.design_space_labels,
            feature_matrix=self.feature_matrix,
        )
        return acds

    @property
    def design_space_structures(self):
        return self._design_space_structures

    @design_space_structures.setter
    def design_space_structures(self, design_space_structures):
        msg = "Please use `update` method to update the design space."
        raise DesignSpaceError(msg)

    @property
    def design_space_labels(self):
        return self._design_space_labels

    @design_space_labels.setter
    def design_space_labels(self, design_space_labels):
        msg = "Please use `update` method to update the design space."
        raise DesignSpaceError(msg)

    @property
    def feature_matrix(self):
        return self._feature_matrix

    @feature_matrix.setter
    def feature_matrix(self, feature_matrix):
        msg = "Please use `update` method to update the design space."
        raise DesignSpaceError(msg)

    @property
    def species_list(self):
        if self._design_space_structures:
            species_list = []
            for s in self.design_space_structures:
                # get all unique species
                found_species = np.unique(s.get_chemical_symbols()).tolist()
                new_species = [
                    spec for spec in found_species if spec not in species_list
                ]
                species_list.extend(new_species)
            return species_list

    def update(
        self,
        structures: List[Atoms] = None,
        labels: Array = None,
        feature_matrix: Array = None,
    ):
        """
        Updates design space given structures and corresponding labels.
        If structure already in design space, the label is updated.

        Parameters
        ----------

        structures:
            List of Atoms objects structures to be added

        labels:
            Corresponding labels to `structures`
        """
        if feature_matrix is not None and structures is not None:
            msg = "Please provide only the feature matrix or list of structures"
            raise DesignSpaceError(msg)

        if (feature_matrix is not None) and (labels is not None):
            if isinstance(feature_matrix, np.ndarray):
                feature_matrix = feature_matrix.tolist()
            assert len(feature_matrix) == len(labels)
            assert len(feature_matrix[0]) == self.feature_matrix.shape[1]
            for i, vec in enumerate(feature_matrix):
                # if structure already in design space, update label
                if vec in self.feature_matrix.tolist():
                    idx = self.feature_matrix.tolist().index(vec)
                    self._design_space_labels[idx] = labels[i]
                else:
                    self._feature_matrix = np.vstack((self._feature_matrix, vec))
                    self._design_space_labels = np.append(
                        self.design_space_labels, labels[i]
                    )
        elif (structures is not None) and (labels is not None):
            assert len(structures) == len(labels)
            assert all(isinstance(struct, Atoms) for struct in structures)
            for i, struct in enumerate(structures):
                # if structure already in design space, update label
                if struct in self.design_space_structures:
                    idx = self.design_space_structures.index(struct)
                    self._design_space_labels[idx] = labels[i]
                # otherwise extend design space
                else:
                    self._design_space_structures.append(struct)
                    self._design_space_labels = np.append(
                        self.design_space_labels, labels[i]
                    )

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        collected_structs = None
        if self.design_space_structures is not None:
            collected_structs = []
            for struct in self.design_space_structures:
                collected_structs.append(atoms_encoder(struct))
        jsonified_labels = [float(x) for x in self.design_space_labels]
        if self.feature_matrix is not None:
            jsonified_feat_mat = self.feature_matrix.tolist()
        else:
            jsonified_feat_mat = None
        return {
            "structures": collected_structs,
            "labels": jsonified_labels,
            "feature_matrix": jsonified_feat_mat,
        }

    def write_json_to_disk(
        self, json_name: str = None, write_location: str = ".",
    ):
        """
        Writes DesignSpace to disk as a json
        """
        collected_jsons = self.to_jsonified_dict()
        # set default json name if needed
        if json_name is None:
            json_name = "acds.json"

        json_path = os.path.join(write_location, json_name)
        with open(json_path, "w") as f:
            json.dump(collected_jsons, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        if (
            all_data.get("structures") is None
            and all_data.get("feature_matrix") is None
        ):
            msg = "Either structures or feature matrix must be provided"
            raise DesignSpaceError(msg)
        if all_data.get("labels") is None:
            msg = "Design space labels must be provided"
            raise DesignSpaceError(msg)
        structures = None
        if all_data.get("structures"):
            try:
                structures = []
                for encoded_atoms in all_data["structures"]:
                    structures.append(atoms_decoder(encoded_atoms))
            except (json.JSONDecodeError, TypeError):
                msg = "Please ensure design space structures encoded using `ase.io.jsonio.encode`"
                raise DesignSpaceError(msg)
        labels = np.array(all_data["labels"])
        if all_data.get("feature_matrix") is not None:
            feature_matrix = np.array(all_data["feature_matrix"])
        else:
            feature_matrix = None
        return DesignSpace(
            design_space_structures=structures,
            design_space_labels=labels,
            feature_matrix=feature_matrix,
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return DesignSpace.from_jsonified_dict(all_data)


class SyntheticDesignSpaceError(Exception):
    pass


class SyntheticDesignSpace:
    def __init__(
        self,
        surface_kind: str = None,
        num_dimensions: int = None,
        discretization_width: float = None,
        noise_scale: float = None,
    ):
        """
        Constructor.

        Wrapper for Olympus surfaces to generate synthetic design spaces.
        The parameter space of the function will be mapped to the unit
        hypercube.

        Parameters
        ----------

        surface_kind:
            String specifying the kind of surface to be used (e.g. AckleyPath).
            Options are available in Olympus

        num_dimensions:
            Dimensionality of the function in value space

        discretization_width:
            Width between points in each axis when discretizing the unit hypercube

        noise_scale:
            Standard deviation of noise to be applied to the values (if desired)
            Default is no noise provided

        """
        self.initialized = False

        self._num_dimensions = 2
        self.num_dimensions = num_dimensions

        self._surface_kind = "AckleyPath"
        self.surface_kind = surface_kind

        self._discretization_width = 0.01
        self.discretization_width = discretization_width

        self._noise_scale = 0
        self.noise_scale = noise_scale

        self.initialize()

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "SyntheticDesignSpace"]
        pt.add_row(["surface kind", self.surface_kind])
        pt.add_row(["number of dimensions", self.num_dimensions])
        pt.add_row(["discretization width", self.discretization_width])
        pt.add_row(["noise scale", self.noise_scale])
        pt.add_row(["initialized?", self.initialized])
        if self.initialized:
            pt.add_row(["total # of discretized points", len(self)])
            surface_max_value = self.surface.maxima
            pt.add_row(["surface maxima", surface_max_value])
            surface_min_value = self.surface.minima
            pt.add_row(["surface minima", surface_min_value])
        pt.max_width = 70
        return str(pt)

    def __len__(self):
        return len(self.feature_matrix)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SyntheticDesignSpace):
            # check that they are the same length
            if len(self) == len(other):
                # check feature matrices are equal
                self_feature_matrix = self.feature_matrix
                o_feature_matrix = other.feature_matrix
                if not np.array_equal(self_feature_matrix, o_feature_matrix):
                    return False

                # check their labels are equal
                self_labels = self.design_space_labels
                o_labels = other.design_space_labels
                return np.array_equal(self_labels, o_labels)
        return False

    def copy(self):
        """
        Returns a copy of the design space
        """
        sds = self.__class__(
            surface_kind=self.surface_kind,
            num_dimensions=self.num_dimensions,
            discretization_width=self.discretization_width,
            noise_scale=self.noise_scale,
        )

        sds._noise = self._noise.copy()
        sds.initialized = self.initialized

        return sds

    @property
    def num_dimensions(self):
        return self._num_dimensions

    @num_dimensions.setter
    def num_dimensions(self, num_dimensions):
        if num_dimensions is not None:
            self._num_dimensions = num_dimensions
            self.initialized = False

    @property
    def discretization_width(self):
        return self._discretization_width

    @discretization_width.setter
    def discretization_width(self, discretization_width):
        if discretization_width is not None:
            self._discretization_width = discretization_width
            self.initialized = False

    @property
    def noise_scale(self):
        return self._noise_scale

    @noise_scale.setter
    def noise_scale(self, noise_scale):
        if noise_scale is not None:
            self._noise_scale = noise_scale
            self.initialized = False

    @property
    def surface_kind(self):
        return self._surface_kind

    @surface_kind.setter
    def surface_kind(self, surface_kind):
        if surface_kind is not None:
            self._surface_kind = surface_kind
            self.initialized = False

    @property
    def noise(self):
        if not self.initialized:
            msg = "Please initialize using the `initialize` method"
            raise SyntheticDesignSpaceError(msg)
        return self._noise

    @property
    def surface(self):
        if not self.initialized:
            msg = "Please initialize using the `initialize` method"
            raise SyntheticDesignSpaceError(msg)
        return self._surface

    @property
    def feature_matrix(self):
        if not self.initialized:
            msg = "Please initialize using the `initialize` method"
            raise SyntheticDesignSpaceError(msg)
        return self._feature_matrix

    @property
    def design_space_labels(self):
        if not self.initialized:
            msg = "Please initialize using the `initialize` method"
            raise SyntheticDesignSpaceError(msg)
        return self._design_space_labels

    def initialize(self):
        """
        Initializes the parameter space, surface, and objective space
        """
        # generate parameter space
        width = self.discretization_width
        axes = [np.arange(0.0, 1.0 + width, width) for _ in range(self.num_dimensions)]
        parameter_space = np.array(list(itertools.product(*axes)))
        self._feature_matrix = parameter_space

        # generate noiseless surface
        self._surface = Surface(kind=self.surface_kind, param_dim=self.num_dimensions)
        objective_space = np.array(self._surface.run(parameter_space)).reshape(-1,)

        # add noise (if desired)
        noise = np.random.normal(scale=self.noise_scale, size=len(parameter_space))
        self._noise = noise

        # store objective space
        self._design_space_labels = objective_space + noise
        self.initialized = True

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        jsonified_labels = None
        if self.design_space_labels is not None:
            jsonified_labels = [float(x) for x in self.design_space_labels]
        jsonified_feat_mat = None
        if self.feature_matrix is not None:
            jsonified_feat_mat = self.feature_matrix.tolist()
        jsonified_noise = None
        if self.noise is not None:
            jsonified_noise = self.noise.tolist()
        return {
            "surface_kind": self.surface_kind,
            "num_dimensions": self.num_dimensions,
            "discretization_width": self.discretization_width,
            "noise_scale": self.noise_scale,
            "initialized": self.initialized,
            "labels": jsonified_labels,
            "feature_matrix": jsonified_feat_mat,
            "noise": jsonified_noise,
        }

    def write_json_to_disk(
        self, json_name: str = None, write_location: str = ".",
    ):
        """
        Writes DesignSpace to disk as a json
        """
        collected_jsons = self.to_jsonified_dict()
        # set default json name if needed
        if json_name is None:
            json_name = "sds.json"

        json_path = os.path.join(write_location, json_name)
        with open(json_path, "w") as f:
            json.dump(collected_jsons, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        sds = SyntheticDesignSpace(
            surface_kind=all_data.get("surface_kind"),
            num_dimensions=all_data.get("num_dimensions"),
            discretization_width=all_data.get("discretization_width"),
            noise_scale=all_data.get("noise_scale"),
        )
        sds.initialized = all_data.get("initialized", False)
        if sds.initialized:
            sds._feature_matrix = np.array(all_data.get("feature_matrix"))
            sds._design_space_labels = np.array(all_data.get("labels"))
            sds._noise = np.array(all_data.get("noise"))
        return sds

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return SyntheticDesignSpace.from_jsonified_dict(all_data)


class AcquisitionFunctionSelectorError(Exception):
    pass


class CyclicAcquisitionStrategy:
    def __init__(
        self,
        exploit_acquisition_function: str = None,
        explore_acquisition_function: str = None,
        fixed_cyclic_strategy: List[int] = None,
        afs_kwargs: Dict[str, int] = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        exploitative_acquisition_function:
            Acquisition function that is more exploitative
            (e.g. MLI)

        explorative_acquisition_function:
            Acquisition function that is more explorative
            (e.g. Random)

        fixed_cyclic_strategy:
            Unit cycle for selecting acquisition function with 0/1 corresponding
            to explore/exploit acquisition function

            e.g. [0, 0, 1] corresponds to alternating pattern of using the explore
            acquisition function twice and exploit acquisition function once

            Default: [0, 1] (represents an alternating pattern between
            the two acquisition functions)

        """

        self._explore_acquisition_function = "Random"
        self.explore_acquisition_function = explore_acquisition_function

        self._exploit_acquisition_function = "MLI"
        self.exploit_acquisition_function = exploit_acquisition_function

        self._fixed_cyclic_strategy = [0, 1]
        self.fixed_cyclic_strategy = fixed_cyclic_strategy

        # other miscellaneous kw arguments
        self.afs_kwargs = afs_kwargs if afs_kwargs else {}
        if "acquisition_function_history" not in self.afs_kwargs:
            self.afs_kwargs.update({"acquisition_function_history": None})

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Fixed Acquisition Strategy"]
        pt.add_row(["exploit_acquisition_function", self.exploit_acquisition_function])
        pt.add_row(["explore_acquisition_function", self.explore_acquisition_function])
        pt.add_row(["fixed_cyclic_strategy", self.fixed_cyclic_strategy])
        if self.fixed_cyclic_strategy == "fixed_cycle":
            pt.add_row(["cyclic schedule", self.fixed_cyclic_strategy])
        return str(pt)

    def copy(self):
        """
        Returns a copy
        """
        fas = self.__class__(
            exploit_acquisition_function=self.exploit_acquisition_function,
            explore_acquisition_function=self.explore_acquisition_function,
            fixed_cyclic_strategy=self.fixed_cyclic_strategy,
        )
        fas.afs_kwargs = copy.deepcopy(self.afs_kwargs)
        return fas

    @property
    def explore_acquisition_function(self):
        return self._explore_acquisition_function

    @explore_acquisition_function.setter
    def explore_acquisition_function(self, explore_acquisition_function):
        if explore_acquisition_function is not None:
            self._explore_acquisition_function = explore_acquisition_function

    @property
    def exploit_acquisition_function(self):
        return self._exploit_acquisition_function

    @exploit_acquisition_function.setter
    def exploit_acquisition_function(self, exploit_acquisition_function):
        if exploit_acquisition_function is not None:
            self._exploit_acquisition_function = exploit_acquisition_function

    @property
    def fixed_cyclic_strategy(self):
        return self._fixed_cyclic_strategy

    @fixed_cyclic_strategy.setter
    def fixed_cyclic_strategy(self, fixed_cyclic_strategy):
        if fixed_cyclic_strategy is not None:
            self._fixed_cyclic_strategy = fixed_cyclic_strategy

    @property
    def acquisition_function_history(self):
        return self.afs_kwargs.get("acquisition_function_history", None)

    def select_acquisition_function(
        self, number_of_labelled_data_pts: int, start_reference: int = 0
    ):
        """
        Selects acquisition function based on number of labelled data points so far
        within a given search

        Parameters
        ----------

        number_of_labelled_data_pts:
            The number of data points for which labels have been calculated

        start_reference:
            Number of labelled data points fixed cycle should start from.
            Useful if changing strategy mid-search

            e.g. for fixed schedule [0, 0, 0, 1] and start_reference=3 will
            start from the last element of this cycle

        """
        aqs = [self.explore_acquisition_function, self.exploit_acquisition_function]

        cycle = self.fixed_cyclic_strategy

        start = start_reference + number_of_labelled_data_pts % len(cycle)

        selector = itertools.islice(itertools.cycle(cycle), start, None)

        acquisition_function_hist = self.afs_kwargs.get("acquisition_function_history")
        if acquisition_function_hist is None:
            acquisition_function_hist = []

        # next acquisition function
        next_aqf = aqs[next(selector)]
        acquisition_function_hist.append(next_aqf)
        self.afs_kwargs.update(
            {"acquisition_function_history": acquisition_function_hist}
        )

        return next_aqf

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        return {
            "exploit_acquisition_function": self.exploit_acquisition_function,
            "explore_acquisition_function": self.explore_acquisition_function,
            "fixed_cyclic_strategy": self.fixed_cyclic_strategy,
            "afs_kwargs": self.afs_kwargs,
        }

    def write_json_to_disk(self, write_location: str = ".", json_name: str = None):
        """
        Writes `CyclicAcquisitionStrategy` to disk as a json
        """
        jsonified_list = self.to_jsonified_dict()

        if json_name is None:
            json_name = "fixed_acquisition_strategy.json"

        json_path = os.path.join(write_location, json_name)

        with open(json_path, "w") as f:
            json.dump(jsonified_list, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        return CyclicAcquisitionStrategy(
            exploit_acquisition_function=all_data.get("exploit_acquisition_function"),
            explore_acquisition_function=all_data.get("explore_acquisition_function"),
            fixed_cyclic_strategy=all_data.get("fixed_cyclic_strategy"),
            afs_kwargs=all_data.get("afs_kwargs", {}),
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return CyclicAcquisitionStrategy.from_jsonified_dict(all_data)


class ProbabilisticAcquisitionStrategy:
    def __init__(
        self,
        exploit_acquisition_function: str = None,
        explore_acquisition_function: str = None,
        explore_probability: float = None,
        random_num_generator: np.random.Generator = None,
        afs_kwargs: Dict[str, int] = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        exploitative_acquisition_function:
            Acquisition function that is more exploitative
            (e.g. MLI)

        explorative_acquisition_function:
            Acquisition function that is more explorative
            (e.g. Random)

        explore_probability:
            Probability of selecting the explore acquisition
            function. Otherwise the exploit acquisition
            function is picked.
            Default: 0.5 (equal chance of picking explore/exploit)

        random_num_generator:
            Numpy random number generator to be used to
            probabilistically choose the acquisition function

        """
        if random_num_generator is None:
            random_num_generator = np.random.default_rng()
        self._random_num_generator = random_num_generator

        self._explore_acquisition_function = "Random"
        self.explore_acquisition_function = explore_acquisition_function

        self._exploit_acquisition_function = "MLI"
        self.exploit_acquisition_function = exploit_acquisition_function

        self._explore_probability = 0.5
        self.explore_probability = explore_probability

        # other miscellaneous kw arguments
        self.afs_kwargs = afs_kwargs if afs_kwargs else {}
        if "acquisition_function_history" not in self.afs_kwargs:
            self.afs_kwargs.update({"acquisition_function_history": None})

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Fixed Acquisition Strategy"]
        pt.add_row(["exploit_acquisition_function", self.exploit_acquisition_function])
        pt.add_row(["explore_acquisition_function", self.explore_acquisition_function])
        pt.add_row(["explore_probability", self.explore_probability])
        return str(pt)

    def copy(self):
        """
        Returns a copy
        """
        fas = self.__class__(
            exploit_acquisition_function=self.exploit_acquisition_function,
            explore_acquisition_function=self.explore_acquisition_function,
            explore_probability=self.explore_probability,
        )
        fas.afs_kwargs = copy.deepcopy(self.afs_kwargs)
        return fas

    @property
    def explore_acquisition_function(self):
        return self._explore_acquisition_function

    @explore_acquisition_function.setter
    def explore_acquisition_function(self, explore_acquisition_function):
        if explore_acquisition_function is not None:
            self._explore_acquisition_function = explore_acquisition_function

    @property
    def exploit_acquisition_function(self):
        return self._exploit_acquisition_function

    @exploit_acquisition_function.setter
    def exploit_acquisition_function(self, exploit_acquisition_function):
        if exploit_acquisition_function is not None:
            self._exploit_acquisition_function = exploit_acquisition_function

    @property
    def explore_probability(self):
        return self._explore_probability

    @explore_probability.setter
    def explore_probability(self, explore_probability):
        if explore_probability is not None:
            self._explore_probability = explore_probability

    @property
    def acquisition_function_history(self):
        return self.afs_kwargs.get("acquisition_function_history", None)

    @property
    def random_num_generator(self):
        return self._random_num_generator

    def select_acquisition_function(self):
        """
        Probabilistically selects either an explorative or exploitative
        acquisition function

        Parameters
        ----------

        number_of_labelled_data_pts:
            The number of data points for which labels have been calculated

        start_reference:
            Number of labelled data points fixed cycle should start from.
            Useful if changing strategy mid-search

            e.g. for fixed schedule [0, 0, 0, 1] and start_reference=3 will
            start from the last element of this cycle

        """
        aqs = [self.explore_acquisition_function, self.exploit_acquisition_function]

        acquisition_function_hist = self.afs_kwargs.get("acquisition_function_history")
        if acquisition_function_hist is None:
            acquisition_function_hist = []

        # next acquisition function
        next_aqf = self.random_num_generator.choice(
            aqs, p=[self.explore_probability, 1.0 - self.explore_probability]
        )
        acquisition_function_hist.append(next_aqf)
        self.afs_kwargs.update(
            {"acquisition_function_history": acquisition_function_hist}
        )

        return next_aqf

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        return {
            "exploit_acquisition_function": self.exploit_acquisition_function,
            "explore_acquisition_function": self.explore_acquisition_function,
            "explore_probability": self.explore_probability,
            "afs_kwargs": self.afs_kwargs,
        }

    def write_json_to_disk(self, write_location: str = ".", json_name: str = None):
        """
        Writes `ProbabilisticAcquisitionStrategy` to disk as a json
        """
        jsonified_list = self.to_jsonified_dict()

        if json_name is None:
            json_name = "proba_acquisition_strategy.json"

        json_path = os.path.join(write_location, json_name)

        with open(json_path, "w") as f:
            json.dump(jsonified_list, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        return ProbabilisticAcquisitionStrategy(
            exploit_acquisition_function=all_data.get("exploit_acquisition_function"),
            explore_acquisition_function=all_data.get("explore_acquisition_function"),
            explore_probability=all_data.get("explore_probability"),
            afs_kwargs=all_data.get("afs_kwargs", {}),
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return ProbabilisticAcquisitionStrategy.from_jsonified_dict(all_data)


class AnnealingAcquisitionStrategy:
    def __init__(
        self,
        exploit_acquisition_function: str = None,
        explore_acquisition_function: str = None,
        anneal_temp: float = None,
        afs_kwargs: Dict[str, int] = None,
    ):
        """
        Constructor.

        This acquisition strategy selects either the
        explore or exploit acquisition function at each
        iteration probabilistically. More explicitly:

        P_i(explore AQF) = exp(-i / ANNEAL_TEMP)
        is the probability of selecting the explore AQF
        with i labelled data points

        P_i(exploit AQF) = 1 - P_i(explore AQF)
        is the probability of selecting the explore AQF
        with i labelled data points

        Parameters
        ----------

        exploitative_acquisition_function:
            Acquisition function that is more exploitative
            (e.g. MLI)

        explorative_acquisition_function:
            Acquisition function that is more explorative
            (e.g. Random)

        anneal_temp:
            Hyperparameter analogous to temperature for
            simulated annealing. The higher this hyperparameter
            is, the more likely the explore AQF is selected
            deeper into the campaign.
            function. Default: 50

        """

        self._explore_acquisition_function = "Random"
        self.explore_acquisition_function = explore_acquisition_function

        self._exploit_acquisition_function = "MLI"
        self.exploit_acquisition_function = exploit_acquisition_function

        self._anneal_temp = 50.0
        self.anneal_temp = anneal_temp

        # other miscellaneous kw arguments
        self.afs_kwargs = afs_kwargs if afs_kwargs else {}
        if "acquisition_function_history" not in self.afs_kwargs:
            self.afs_kwargs.update({"acquisition_function_history": None})

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Annealing Acquisition Strategy"]
        pt.add_row(["exploit_acquisition_function", self.exploit_acquisition_function])
        pt.add_row(["explore_acquisition_function", self.explore_acquisition_function])
        pt.add_row(["anneal_temp", self.anneal_temp])
        return str(pt)

    def copy(self):
        """
        Returns a copy
        """
        aas = self.__class__(
            exploit_acquisition_function=self.exploit_acquisition_function,
            explore_acquisition_function=self.explore_acquisition_function,
            anneal_temp=self.anneal_temp,
        )
        aas.afs_kwargs = copy.deepcopy(self.afs_kwargs)
        return aas

    @property
    def explore_acquisition_function(self):
        return self._explore_acquisition_function

    @explore_acquisition_function.setter
    def explore_acquisition_function(self, explore_acquisition_function):
        if explore_acquisition_function is not None:
            self._explore_acquisition_function = explore_acquisition_function

    @property
    def exploit_acquisition_function(self):
        return self._exploit_acquisition_function

    @exploit_acquisition_function.setter
    def exploit_acquisition_function(self, exploit_acquisition_function):
        if exploit_acquisition_function is not None:
            self._exploit_acquisition_function = exploit_acquisition_function

    @property
    def anneal_temp(self):
        return self._anneal_temp

    @anneal_temp.setter
    def anneal_temp(self, anneal_temp):
        if anneal_temp is not None:
            self._anneal_temp = anneal_temp

    @property
    def acquisition_function_history(self):
        return self.afs_kwargs.get("acquisition_function_history", None)

    def select_acquisition_function(
        self, number_of_labelled_data_pts: int, rng: np.random.Generator = None
    ):
        """
        Selects acquisition function based on number of labelled data points so far
        within a given search

        Parameters
        ----------

        number_of_labelled_data_pts:
            The number of data points for which labels have been calculated

        rng:
            Numpy random number generator (for testing)

        """
        aqs = [self.explore_acquisition_function, self.exploit_acquisition_function]

        p_explore = np.exp(-number_of_labelled_data_pts / self.anneal_temp)

        acquisition_function_hist = self.afs_kwargs.get("acquisition_function_history")
        if acquisition_function_hist is None:
            acquisition_function_hist = []

        # next acquisition function
        if rng is None:
            rng = np.random.default_rng()
        next_aqf = rng.choice(aqs, p=(p_explore, 1 - p_explore), shuffle=False)
        acquisition_function_hist.append(next_aqf)
        self.afs_kwargs.update(
            {"acquisition_function_history": acquisition_function_hist}
        )

        return next_aqf

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        return {
            "exploit_acquisition_function": self.exploit_acquisition_function,
            "explore_acquisition_function": self.explore_acquisition_function,
            "anneal_temp": self.anneal_temp,
            "afs_kwargs": self.afs_kwargs,
        }

    def write_json_to_disk(self, write_location: str = ".", json_name: str = None):
        """
        Writes `AnnealingAcquisitionStrategy` to disk as a json
        """
        jsonified_list = self.to_jsonified_dict()

        if json_name is None:
            json_name = "annealing_acquisition_strategy.json"

        json_path = os.path.join(write_location, json_name)

        with open(json_path, "w") as f:
            json.dump(jsonified_list, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        return AnnealingAcquisitionStrategy(
            exploit_acquisition_function=all_data.get("exploit_acquisition_function"),
            explore_acquisition_function=all_data.get("explore_acquisition_function"),
            anneal_temp=all_data.get("anneal_temp"),
            afs_kwargs=all_data.get("afs_kwargs", {}),
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return AnnealingAcquisitionStrategy.from_jsonified_dict(all_data)


class ThresholdAcquisitionStrategy:
    def __init__(
        self,
        exploit_acquisition_function: str = None,
        explore_acquisition_function: str = None,
        uncertainty_cutoff: float = None,
        min_fraction_less_than_unc_cutoff: float = None,
        afs_kwargs: Dict[str, int] = None,
    ):
        """
        Constructor.

        This acquisition strategy selects either the
        explore or exploit acquisition function at each
        step based on the fraction of candidate systems
        whose uncertainties fall below a specified
        cutoff.

        When the fraction of candidates whose uncertainties
        are within the cutoff is at the specified minimum,
        the exploit AQF is picked. Otherwise the explore
        AQF is used.

        Mathematically:

        AQF = { explore AQF if n(sigma_D < epsilon) / N > k
              { exploit AQF otherwise
        where n is the number of candidates with uncertainty less than
        epsilon, N is the total number of candidates and k is the desired
        minimum fraction

        Parameters
        ----------

        exploitative_acquisition_function:
            Acquisition function that is more exploitative
            (e.g. MLI)

        explorative_acquisition_function:
            Acquisition function that is more explorative
            (e.g. Random)

        uncertainty_cutoff:
            Maximum allowed uncertainty that is acceptable

        min_fraction_less_than_unc_cutoff:
            Desired fraction of candidate systems whose uncertainties
            are less than the cutoff defined by `uncertainty_cutoff`

        """

        self._explore_acquisition_function = "Random"
        self.explore_acquisition_function = explore_acquisition_function

        self._exploit_acquisition_function = "MLI"
        self.exploit_acquisition_function = exploit_acquisition_function

        self._uncertainty_cutoff = 0.1
        self.uncertainty_cutoff = uncertainty_cutoff

        self._min_fraction_less_than_unc_cutoff = 0.75
        self.min_fraction_less_than_unc_cutoff = min_fraction_less_than_unc_cutoff

        # other miscellaneous kw arguments
        self.afs_kwargs = afs_kwargs if afs_kwargs else {}
        if "acquisition_function_history" not in self.afs_kwargs:
            self.afs_kwargs.update({"acquisition_function_history": None})

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Annealing Acquisition Strategy"]
        pt.add_row(["exploit_acquisition_function", self.exploit_acquisition_function])
        pt.add_row(["explore_acquisition_function", self.explore_acquisition_function])
        pt.add_row(["uncertainty cutoff", self.uncertainty_cutoff])
        pt.add_row(
            [
                "fraction less than uncertainty cutoff",
                self.min_fraction_less_than_unc_cutoff,
            ]
        )
        return str(pt)

    def copy(self):
        """
        Returns a copy
        """
        tas = self.__class__(
            exploit_acquisition_function=self.exploit_acquisition_function,
            explore_acquisition_function=self.explore_acquisition_function,
            uncertainty_cutoff=self.uncertainty_cutoff,
            min_fraction_less_than_unc_cutoff=self.min_fraction_less_than_unc_cutoff,
        )
        tas.afs_kwargs = copy.deepcopy(self.afs_kwargs)
        return tas

    @property
    def explore_acquisition_function(self):
        return self._explore_acquisition_function

    @explore_acquisition_function.setter
    def explore_acquisition_function(self, explore_acquisition_function):
        if explore_acquisition_function is not None:
            self._explore_acquisition_function = explore_acquisition_function

    @property
    def exploit_acquisition_function(self):
        return self._exploit_acquisition_function

    @exploit_acquisition_function.setter
    def exploit_acquisition_function(self, exploit_acquisition_function):
        if exploit_acquisition_function is not None:
            self._exploit_acquisition_function = exploit_acquisition_function

    @property
    def uncertainty_cutoff(self):
        return self._uncertainty_cutoff

    @uncertainty_cutoff.setter
    def uncertainty_cutoff(self, uncertainty_cutoff):
        if uncertainty_cutoff is not None:
            self._uncertainty_cutoff = uncertainty_cutoff

    @property
    def min_fraction_less_than_unc_cutoff(self):
        return self._min_fraction_less_than_unc_cutoff

    @min_fraction_less_than_unc_cutoff.setter
    def min_fraction_less_than_unc_cutoff(self, min_fraction_less_than_unc_cutoff):
        if min_fraction_less_than_unc_cutoff is not None:
            self._min_fraction_less_than_unc_cutoff = min_fraction_less_than_unc_cutoff

    @property
    def acquisition_function_history(self):
        return self.afs_kwargs.get("acquisition_function_history", None)

    def select_acquisition_function(self, uncertainties: Array, allowed_idx: Array):
        """
        Selects acquisition function based on number of labelled data points so far
        within a given search

        Parameters
        ----------

        number_of_labelled_data_pts:
            The number of data points for which labels have been calculated

        """
        acquisition_function_hist = self.afs_kwargs.get("acquisition_function_history")
        if acquisition_function_hist is None:
            acquisition_function_hist = []

        cand_uncs = uncertainties[allowed_idx]
        fraction_certain = sum(cand_uncs < self.uncertainty_cutoff) / len(cand_uncs)

        # next acquisition function
        if fraction_certain > self.min_fraction_less_than_unc_cutoff:
            next_aqf = self.exploit_acquisition_function

        else:
            next_aqf = self.explore_acquisition_function

        acquisition_function_hist.append(next_aqf)
        self.afs_kwargs.update(
            {"acquisition_function_history": acquisition_function_hist}
        )

        return next_aqf

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        return {
            "exploit_acquisition_function": self.exploit_acquisition_function,
            "explore_acquisition_function": self.explore_acquisition_function,
            "uncertainty_cutoff": self.uncertainty_cutoff,
            "min_fraction_less_than_unc_cutoff": self.min_fraction_less_than_unc_cutoff,
            "afs_kwargs": self.afs_kwargs,
        }

    def write_json_to_disk(self, write_location: str = ".", json_name: str = None):
        """
        Writes `ThresholdAcquisitionStrategy` to disk as a json
        """
        jsonified_list = self.to_jsonified_dict()

        if json_name is None:
            json_name = "threshold_acquisition_strategy.json"

        json_path = os.path.join(write_location, json_name)

        with open(json_path, "w") as f:
            json.dump(jsonified_list, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        return ThresholdAcquisitionStrategy(
            exploit_acquisition_function=all_data.get("exploit_acquisition_function"),
            explore_acquisition_function=all_data.get("explore_acquisition_function"),
            uncertainty_cutoff=all_data.get("uncertainty_cutoff"),
            min_fraction_less_than_unc_cutoff=all_data.get(
                "min_fraction_less_than_unc_cutoff"
            ),
            afs_kwargs=all_data.get("afs_kwargs", {}),
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return ThresholdAcquisitionStrategy.from_jsonified_dict(all_data)


class CandidateSelectorError(Exception):
    pass


class CandidateSelector:
    def __init__(
        self,
        acquisition_function: str = None,
        acquisition_strategy: Union[
            CyclicAcquisitionStrategy,
            AnnealingAcquisitionStrategy,
            ThresholdAcquisitionStrategy,
            ProbabilisticAcquisitionStrategy,
        ] = None,
        num_candidates_to_pick: int = None,
        target_window: Array = None,
        include_hhi: bool = None,
        hhi_type: str = "production",
        include_segregation_energies: bool = None,
        segregation_energy_data_source: str = None,
        beta: float = None,
        epsilon: float = None,
        delta: float = None,
        eta: float = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        acquisition_function:
            Acquisition function to be used to select the next candidates
            Options
            - MLI: maximum likelihood of improvement (default)
            - Random
            - MU: maximum uncertainty
            - MEI: maximum expected improvement
            - UCB: upper confidence bound
            - LCB: lower confidence bound
            - GP-UCB: upper confidence bound with variable uncertainty weighting

                AQ_j = mu_j + sqrt(beta) * sigma_j for candidate j
                where beta = 2 log (|D| t**2 pi ** 2 / (6 delta))
                (D: input domain, t: number of labelled points, delta: hyperparameter)

                For more details see:
                Srinivas, et. al. arXiv:0912.3995 (2010)
            - LCBAdaptive: adaptive lower confidence bound described by

                AQ_j = mu_j - epsilon^n * beta * sigma_j
                for candidate j.

                For more details see:
                Siemenn, et. al., npj Comp. Mater, 9, 79 (2023)
            - EIAbrupt: adaptive function consisting of EI and LCB as follows:

                AQ_j = {MEI if |Delta(y_{n-3}, ..., y_{n})| < eta
                       {LCB else
                for candidate j.

                For more details see:
                Siemenn, et. al., npj Comp. Mater, 9, 79 (2023)

        acquisition_strategy:
            Strategy towards selecting acquisition function at each
            iteration. Supercedes `acquisition_function` if provided

        num_candidates_to_pick:
            Number of candidates to choose from the dataset

        target_window:
            Target window that the candidate should ideally fall within

        include_hhi:
            Whether HHI scores should be used to weight aq scores

        hhi_type:
            Type of HHI index to be used for weighting
            Options
            - production (default)
            - reserves

        include_segregation_energies:
            Whether segregation energies should be used to weight aq scores

        segregation_energy_data_source:
            Which tabulated data should the segregation energies be pulled from.
            Options:
            - "raban1999": A.V. Raban, et. al. Phys. Rev. B 59, 15990 (1999)
            - "rao2020": K. K. Rao, et. al. Topics in Catalysis volume 63, pages728-741 (2020)

        beta:
            If using UCB/LCB/LCBAdaptive as the acquisition function,
            this parameter determines the weight of the uncertainty term

        epsilon:
            If using LCBAdaptive as the acquisition function,
            this is an additonal parameter for weighting of the uncertainty term
            as a function of iteration count

        delta:
            If using GP-UCB as the acquisition function, this is an additional
            parameter in the beta uncertainty weighting term.

        eta:
            If using EIAbrupt as the acquisition function, this hyperparameter
            is used to pick either MEI or LCB based on gradient across last 3
            iterations
        """
        self._acquisition_function = "Random"
        self.acquisition_function = acquisition_function

        self._acquisition_strategy = None
        self.acquisition_strategy = acquisition_strategy

        self._num_candidates_to_pick = 1
        self.num_candidates_to_pick = num_candidates_to_pick

        self._target_window = None
        self.target_window = target_window

        self._include_hhi = False
        self.include_hhi = include_hhi

        self._hhi_type = "production"
        self.hhi_type = hhi_type

        self._include_segregation_energies = False
        self.include_segregation_energies = include_segregation_energies

        self._segregation_energy_data_source = "raban1999"
        self.segregation_energy_data_source = segregation_energy_data_source

        self._beta = 3 if self.acquisition_function == "LCBAdaptive" else 0.1
        self.beta = beta

        self._epsilon = 0.9
        self.epsilon = epsilon

        self._delta = 0.1
        self.delta = delta

        self._eta = 0.0
        self.eta = eta

    @property
    def acquisition_function(self):
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function):
        if acquisition_function is not None:
            if acquisition_function in [
                "MLI",
                "MU",
                "Random",
                "MEI",
                "UCB",
                "LCB",
                "GP-UCB",
                "LCBAdaptive",
                "EIAbrupt",
            ]:
                self._acquisition_function = acquisition_function
            else:
                msg = f"Unrecognized acquisition function {acquisition_function}\
                     Please select one of 'MLI', 'MU', 'Random', 'UCB', 'LCB', 'LCBAdaptive',\
                     'GP-UCB', 'EIAbrupt'"
                raise CandidateSelectorError(msg)

    @property
    def acquisition_strategy(self):
        return self._acquisition_strategy

    @acquisition_strategy.setter
    def acquisition_strategy(self, acquisition_strategy):
        if acquisition_strategy is not None and isinstance(
            acquisition_strategy,
            (
                CyclicAcquisitionStrategy,
                AnnealingAcquisitionStrategy,
                ThresholdAcquisitionStrategy,
                ProbabilisticAcquisitionStrategy,
            ),
        ):
            self._acquisition_strategy = acquisition_strategy
        elif acquisition_strategy is None:
            pass
        else:
            msg = f"Unrecognized acquisition strategy {acquisition_strategy}"
            raise CandidateSelectorError(msg)

    @property
    def num_candidates_to_pick(self):
        return self._num_candidates_to_pick

    @num_candidates_to_pick.setter
    def num_candidates_to_pick(self, num_candidates_to_pick):
        if num_candidates_to_pick is not None:
            self._num_candidates_to_pick = num_candidates_to_pick

    @property
    def target_window(self):
        return self._target_window

    @target_window.setter
    def target_window(self, target_window):
        if target_window is not None:
            assert len(target_window) == 2
            # ensure not setting infinite window
            if np.array_equal(target_window, np.array([-np.inf, np.inf])):
                msg = "Cannot have an inifite target window"
                raise CandidateSelectorError(msg)
            # sorts window bounds so min is first entry
            sorted_window = np.sort(target_window)
            self._target_window = sorted_window

    @property
    def include_hhi(self):
        return self._include_hhi

    @include_hhi.setter
    def include_hhi(self, include_hhi):
        if include_hhi is not None:
            self._include_hhi = include_hhi

    @property
    def hhi_type(self):
        return self._hhi_type

    @hhi_type.setter
    def hhi_type(self, hhi_type):
        if hhi_type is not None:
            if hhi_type in ["production", "reserves"]:
                self._hhi_type = hhi_type
            else:
                msg = f"Unrecognized HHI type {hhi_type}.\
                     Please select one of 'production' or 'reserves'"
                raise CandidateSelectorError(msg)

    @property
    def include_segregation_energies(self):
        return self._include_segregation_energies

    @include_segregation_energies.setter
    def include_segregation_energies(self, include_segregation_energies):
        if include_segregation_energies is not None:
            self._include_segregation_energies = include_segregation_energies

    @property
    def segregation_energy_data_source(self):
        return self._segregation_energy_data_source

    @segregation_energy_data_source.setter
    def segregation_energy_data_source(self, segregation_energy_data_source):
        if segregation_energy_data_source is not None:
            if segregation_energy_data_source in ["raban1999", "rao2020"]:
                self._segregation_energy_data_source = segregation_energy_data_source
            else:
                msg = f"Unrecognized segregation energy data source\
                    {segregation_energy_data_source}.\
                     Please select one of 'raban1999' or 'rao2020'"
                raise CandidateSelectorError(msg)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is not None:
            self._beta = beta

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon is not None:
            self._epsilon = epsilon

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        if delta is not None:
            self._delta = delta

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        if eta is not None:
            self._eta = eta

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Candidate Selector"]
        pt.add_row(["# of candidates to pick", self.num_candidates_to_pick])
        pt.add_row(["target window", self.target_window])
        pt.add_row(["include hhi?", self.include_hhi])
        if self.include_hhi:
            pt.add_row(["hhi type", self.hhi_type])
        pt.add_row(["include segregation energies?", self.include_segregation_energies])
        if self.include_segregation_energies:
            pt.add_row(
                [
                    "segregation energies data source",
                    self.segregation_energy_data_source,
                ]
            )
        if self.acquisition_strategy is not None:
            aqs = [
                self.acquisition_strategy.exploit_acquisition_function,
                self.acquisition_strategy.explore_acquisition_function,
            ]
            if aqs[0] in ["UCB", "LCB", "LCBAdaptive"] or aqs[1] in [
                "UCB",
                "LCB",
                "LCBAdaptive",
            ]:
                pt.add_row(["beta", self.beta])
            if "LCBAdaptive" in aqs:
                pt.add_row(["epsilon", self.epsilon])
            return str(pt) + "\n" + str(self.acquisition_strategy)
        else:
            pt.add_row(["acquisition function", self.acquisition_function])
            if self.acquisition_function in ["UCB", "LCB", "LCBAdaptive"]:
                pt.add_row(["beta", self.beta])
            elif self.acquisition_function == "GP-UCB":
                pt.add_row(["delta", self.delta])
            if self.acquisition_function == "LCBAdaptive":
                pt.add_row(["epsilon", self.epsilon])
        pt.max_width = 70
        return str(pt)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CandidateSelector):
            for prop in [
                "acquisition_function",
                "num_candidates_to_pick",
                "include_hhi",
                "hhi_type",
                "include_segregation_energies",
                "segregation_energy_data_source",
            ]:
                if getattr(self, prop) != getattr(other, prop):
                    return False
            if not np.isclose(self.beta, other.beta) or not np.isclose(
                self.epsilon, other.epsilon
            ):
                return False
            return np.array_equal(self.target_window, other.target_window)
        return False

    def copy(self):
        """
        Returns a copy of the CandidateSelector
        """
        cs = self.__class__(
            acquisition_function=self.acquisition_function,
            acquisition_strategy=self.acquisition_strategy,
            num_candidates_to_pick=self.num_candidates_to_pick,
            target_window=self.target_window,
            include_hhi=self.include_hhi,
            hhi_type=self.hhi_type,
            include_segregation_energies=self.include_segregation_energies,
        )
        return cs

    def choose_candidate(
        self,
        design_space: DesignSpace,
        allowed_idx: Array = None,
        predictions: Array = None,
        uncertainties: Array = None,
        number_of_labelled_data_pts: int = None,
        cand_label_hist: Array = None,
        **kwargs,
    ):
        """
        Choose the next candidate(s) from a design space

        Parameters
        ----------

        design_space:
            DesignSpace where candidates will be selected from

        allowed_idx:
            Allowed indices that the selector can choose from when making a recommendation
            Defaults to only choosing from systems with `np.nan` labels if a `DesignSpace`
            with unknown labels is provided. Otherwise, all structures are considered

        predictions:
            Predictions for all structures in the DesignSpace

        uncertainties:
            Uncertainties for all structures in the DesignSpace

        number_of_labelled_data_pts:
            The number of data points for which labels have been calculated

        Returns
        -------

        parent_idx:
            Index/indices of the selected candidates

        max_scores:
            Maximum scores (corresponding to the selected candidates)

        aq_scores:
            Calculated scores using `acquisition_function` for the entire DesignSpace
        """
        ds_size = len(design_space)

        if number_of_labelled_data_pts is None:
            number_of_labelled_data_pts = ds_size - sum(
                np.isnan(design_space.design_space_labels)
            )

        if allowed_idx is None:
            if True in np.isnan(design_space.design_space_labels):
                allowed_idx = np.where(np.isnan(design_space.design_space_labels))[0]
            else:
                allowed_idx = np.ones(ds_size, dtype=bool)

        hhi_scores = np.ones(ds_size)
        if self.include_hhi:
            if not design_space.design_space_structures:
                msg = "At present HHI can only be included when structures are provided"
                raise CandidateSelectorError(msg)
            hhi_scores = calculate_hhi_scores(
                design_space.design_space_structures, self.hhi_type
            )

        segreg_energy_scores = np.ones(ds_size)
        if self.include_segregation_energies:
            if not design_space.design_space_structures:
                msg = "At present segregation energies can only be included when structures\
                      are provided"
                raise CandidateSelectorError(msg)
            segreg_energy_scores = calculate_segregation_energy_scores(
                design_space.design_space_structures
            )

        # use acquisition strategy to select acquisition function
        if isinstance(self.acquisition_strategy, ProbabilisticAcquisitionStrategy):
            aq = self.acquisition_strategy.select_acquisition_function()
        elif isinstance(self.acquisition_strategy, CyclicAcquisitionStrategy):
            aq = self.acquisition_strategy.select_acquisition_function(
                number_of_labelled_data_pts=number_of_labelled_data_pts, **kwargs
            )
        elif isinstance(self.acquisition_strategy, AnnealingAcquisitionStrategy):
            aq = self.acquisition_strategy.select_acquisition_function(
                number_of_labelled_data_pts=number_of_labelled_data_pts, **kwargs
            )
        elif isinstance(self.acquisition_strategy, ThresholdAcquisitionStrategy):
            aq = self.acquisition_strategy.select_acquisition_function(
                uncertainties=uncertainties, allowed_idx=allowed_idx
            )
        else:
            # acquisition function already directly specified
            aq = self.acquisition_function

        # pick either MEI or LCB using EIAbrupt
        if aq == "EIAbrupt":
            if cand_label_hist is None:
                msg = "Uncertainty history must be provided for EIAbrupt"
                raise CandidateSelectorError(msg)
            elif (
                len(cand_label_hist) < 3
                or not abs(min(np.gradient(cand_label_hist[-3:]))) > self.eta
            ):
                # stuck in a well or very early in search
                aq = "LCB"
            else:
                aq = "MEI"

        # calculate scores from selected acquisition function
        if aq == "Random":
            raw_scores = np.random.choice(ds_size, size=ds_size, replace=False)

        elif aq == "MU":
            if uncertainties is None:
                msg = "For 'MU', the uncertainties must be supplied"
                raise CandidateSelectorError(msg)
            raw_scores = uncertainties.copy()

        elif aq in ["MLI", "MEI", "UCB", "LCB", "LCBAdaptive", "GP-UCB"]:
            if uncertainties is None or predictions is None:
                msg = f"For {aq}, both uncertainties and predictions must be supplied"
                raise CandidateSelectorError(msg)
            target_window = self.target_window
            if aq == "MLI":
                raw_scores = np.array(
                    [
                        get_overlap_score(
                            mean, std, x2=target_window[1], x1=target_window[0]
                        )
                        for mean, std in zip(predictions, uncertainties)
                    ]
                )
            elif aq == "MEI":
                if sum(np.isinf(target_window)) < 1:
                    msg = f"Finite target window not currently supported for {aq}"
                    raise CandidateSelectorError(msg)
                max_or_min = (-1) ** (np.where(~np.isinf(target_window))[0][0])
                if max_or_min > 0:
                    # maximization problem
                    best = target_window[0]
                    z = (predictions - best) / uncertainties
                else:
                    # minimization problem
                    best = target_window[1]
                    z = (best - predictions) / uncertainties
                norm_dist = stats.norm()
                raw_scores = uncertainties * (z * norm_dist.cdf(z) + norm_dist.pdf(z))
            elif aq in ["UCB", "LCB", "LCBAdaptive", "GP-UCB"]:
                if sum(np.isinf(target_window)) < 1:
                    msg = f"Finite target window not currently supported for {aq}"
                    raise CandidateSelectorError(msg)
                max_or_min = (-1) ** (np.where(~np.isinf(target_window))[0][0])
                if aq == "UCB":
                    raw_scores = predictions + self.beta * uncertainties
                elif aq == "LCB":
                    raw_scores = predictions - self.beta * uncertainties
                elif aq == "LCBAdaptive":
                    if number_of_labelled_data_pts is None:
                        msg = "For LCBAdaptive the iteration count must be provided"
                        raise CandidateSelectorError(msg)
                    raw_scores = (
                        predictions
                        - self.epsilon ** number_of_labelled_data_pts
                        * np.sqrt(self.beta)
                        * uncertainties
                    )
                elif aq == "GP-UCB":
                    if number_of_labelled_data_pts is None:
                        msg = f"For {aq} the iteration count must be provided"
                        raise CandidateSelectorError(msg)
                    D = len(design_space)
                    beta = 2.0 * np.log(
                        D
                        * (number_of_labelled_data_pts + 1) ** 2
                        * np.pi ** 2
                        / 6.0
                        / self.delta
                    )
                    raw_scores = predictions + np.sqrt(beta) * uncertainties
                raw_scores = raw_scores * max_or_min

        aq_scores = raw_scores * hhi_scores * segreg_energy_scores

        next_idx = np.argsort(aq_scores[allowed_idx])[-self.num_candidates_to_pick :]
        sorted_array = aq_scores[allowed_idx][next_idx]
        max_scores = list(sorted_array[-self.num_candidates_to_pick :])
        parent_idx = np.arange(aq_scores.shape[0])[allowed_idx][next_idx]

        return parent_idx, max_scores, aq_scores

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        target_window = self.target_window
        if target_window is not None:
            target_window = [float(x) for x in target_window]
        return {
            "acquisition_function": self.acquisition_function,
            "acquisition_strategy": self.acquisition_strategy.to_jsonified_dict()
            if self.acquisition_strategy
            else None,
            "num_candidates_to_pick": self.num_candidates_to_pick,
            "target_window": target_window,
            "include_hhi": self.include_hhi,
            "hhi_type": self.hhi_type,
            "include_segregation_energies": self.include_segregation_energies,
            "segregation_energy_data_source": self.segregation_energy_data_source,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "eta": self.eta,
        }

    def write_json_to_disk(
        self, json_name: str = None, write_location: str = ".",
    ):
        """
        Writes CandidateSelector to disk as a json
        """
        collected_jsons = self.to_jsonified_dict()
        # set default json name if needed
        if json_name is None:
            json_name = "candidate_selector.json"

        json_path = os.path.join(write_location, json_name)
        with open(json_path, "w") as f:
            json.dump(collected_jsons, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        target_window = all_data.get("target_window")
        if target_window is not None:
            target_window = np.array(target_window)
        cas = all_data.get("acquisition_strategy")
        if cas is not None:
            cas = CyclicAcquisitionStrategy.from_jsonified_dict(cas)
        return CandidateSelector(
            acquisition_function=all_data.get("acquisition_function"),
            acquisition_strategy=cas,
            num_candidates_to_pick=all_data.get("num_candidates_to_pick"),
            target_window=target_window,
            include_hhi=all_data.get("include_hhi"),
            hhi_type=all_data.get("hhi_type"),
            include_segregation_energies=all_data.get("include_segregation_energies"),
            segregation_energy_data_source=all_data.get(
                "segregation_energy_data_source"
            ),
            beta=all_data.get("beta"),
            epsilon=all_data.get("epsilon"),
            delta=all_data.get("delta"),
            eta=all_data.get("eta"),
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return CandidateSelector.from_jsonified_dict(all_data)


class SequentialLearnerError(Exception):
    pass


# TODO: "kwargs" -> "options"?
class SequentialLearner:
    def __init__(
        self,
        design_space: DesignSpace,
        predictor: Predictor = None,
        candidate_selector: CandidateSelector = None,
        fixed_target: bool = True,
        sl_kwargs: Dict[str, int] = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        design_space:
            DesignSpace that is being explored

        predictor:
            Predictor used for training and predicting on the desired property

        candidate_selector:
            CandidateSelector used for calculating scores and selecting candidates
            for each iteration

        fixed_target:
            Whether to keep the target bounds fixed or move based on identified
            candidates. E.g. in a maximization problem keep target value fixed or
            change according to maximum observed value in search so far.
            Currently only implemented for maximization and minimization problems
            Defaults to having a fixed target
        """
        # TODO: move predefined attributes (train_idx, candidate_idxs) to a
        # different container (not kwargs)

        self._design_space = None
        self.design_space = design_space.copy()

        self._predictor = Predictor()
        self.predictor = predictor

        self._candidate_selector = CandidateSelector()
        self.candidate_selector = candidate_selector

        self._fixed_target = True
        self.fixed_target = fixed_target

        # other miscellaneous kw arguments
        self.sl_kwargs = sl_kwargs if sl_kwargs else {}

        # variables that need to be propagated through the SL process
        if "iteration_count" not in self.sl_kwargs:
            self.sl_kwargs.update({"iteration_count": 0})
        if "train_idx" not in self.sl_kwargs:
            self.sl_kwargs.update({"train_idx": None})
        if "train_idx_history" not in self.sl_kwargs:
            self.sl_kwargs.update({"train_idx_history": None})
        if "predictions" not in self.sl_kwargs:
            self.sl_kwargs.update({"predictions": None})
        if "predictions_history" not in self.sl_kwargs:
            self.sl_kwargs.update({"predictions_history": None})
        if "uncertainties" not in self.sl_kwargs:
            self.sl_kwargs.update({"uncertainties": None})
        if "uncertainties_history" not in self.sl_kwargs:
            self.sl_kwargs.update({"uncertainties_history": None})
        if "candidate_indices" not in self.sl_kwargs:
            self.sl_kwargs.update({"candidate_indices": None})
        if "candidate_index_history" not in self.sl_kwargs:
            self.sl_kwargs.update({"candidate_index_history": None})
        if "acquisition_scores" not in self.sl_kwargs:
            self.sl_kwargs.update({"acquisition_scores": None})
        if "acquisition_score_history" not in self.sl_kwargs:
            self.sl_kwargs.update({"acquisition_score_history": None})
        if "target_window_history" not in self.sl_kwargs:
            self.sl_kwargs.update({"target_window_history": None})

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Sequential Learner"]
        pt.add_row(["iteration count", self.iteration_count])
        if self.candidate_structures is not None:
            cand_formulas = [
                s.get_chemical_formula() for s in self.candidate_structures
            ]
        else:
            cand_formulas = None
        pt.add_row(["next candidate system structures", cand_formulas])
        pt.add_row(["next candidate system indices", self.candidate_indices])
        return (
            str(pt)
            + "\n"
            + str(self.candidate_selector)
            + "\n"
            + str(self.design_space)
            + "\n"
            + str(self.predictor)
        )

    @property
    def design_space(self):
        return self._design_space

    @design_space.setter
    def design_space(self, design_space):
        if design_space is not None and isinstance(design_space, DesignSpace):
            self._design_space = design_space

    @property
    def predictor(self):
        return self._predictor

    @predictor.setter
    def predictor(self, predictor):
        if predictor is not None and isinstance(predictor, Predictor):
            feat = predictor.featurizer.copy()
            feat.design_space_structures = self.design_space.design_space_structures
            self._predictor = Predictor(regressor=predictor.regressor, featurizer=feat)

    @property
    def candidate_selector(self):
        return self._candidate_selector

    @candidate_selector.setter
    def candidate_selector(self, candidate_selector):
        if candidate_selector is not None and isinstance(
            candidate_selector, CandidateSelector
        ):
            self._candidate_selector = candidate_selector.copy()

    @property
    def fixed_target(self):
        return self._fixed_target

    @fixed_target.setter
    def fixed_target(self, fixed_target):
        if fixed_target is not None and isinstance(fixed_target, bool):
            self._fixed_target = fixed_target

    @property
    def iteration_count(self):
        return self.sl_kwargs.get("iteration_count", 0)

    @property
    def train_idx(self):
        return self.sl_kwargs.get("train_idx")

    @property
    def train_idx_history(self):
        return self.sl_kwargs.get("train_idx_history", None)

    @property
    def predictions(self):
        return self.sl_kwargs.get("predictions")

    @property
    def uncertainties(self):
        return self.sl_kwargs.get("uncertainties")

    @property
    def candidate_indices(self):
        return self.sl_kwargs.get("candidate_indices")

    @property
    def acquisition_scores(self):
        return self.sl_kwargs.get("acquisition_scores", None)

    @property
    def acquisition_score_history(self):
        return self.sl_kwargs.get("acquisition_score_history")

    @property
    def candidate_structures(self):
        idxs = self.candidate_indices
        if idxs is not None and self.design_space.design_space_structures is not None:
            return [self.design_space.design_space_structures[i] for i in idxs]

    @property
    def candidate_index_history(self):
        return self.sl_kwargs.get("candidate_index_history", None)

    @property
    def predictions_history(self):
        return self.sl_kwargs.get("predictions_history", None)

    @property
    def uncertainties_history(self):
        return self.sl_kwargs.get("uncertainties_history", None)

    @property
    def target_window_history(self):
        return self.sl_kwargs.get("target_window_history", None)

    def copy(self):
        """
        Returns a copy
        """
        acsl = self.__class__(
            design_space=self.design_space,
            predictor=self.predictor,
            candidate_selector=self.candidate_selector,
        )
        acsl.sl_kwargs = copy.deepcopy(self.sl_kwargs)
        return acsl

    def iterate(self):
        """Runs the next iteration of sequential learning.

        This process consists of:
        - retraining the predictor
        - adjusting the target bounds if `fixed_target` is False
        - predicting candidate properties and calculating candidate scores (if
        fully explored returns None)
        - selecting the next batch of candidates for objective evaluation (if
        fully explored returns None)
        """

        # dstructs = self.design_space.design_space_structures
        dsystems = (
            self.design_space.feature_matrix
            if self.design_space.feature_matrix is not None
            else self.design_space.design_space_structures
        )
        dlabels = self.design_space.design_space_labels

        mask_nans = ~np.isnan(dlabels)
        if isinstance(dsystems, np.ndarray):
            masked_systems = dsystems[np.where(mask_nans)]
        else:
            masked_systems = [
                struct for i, struct in enumerate(dsystems) if mask_nans[i]
            ]
        masked_labels = dlabels[np.where(mask_nans)]

        self.predictor.fit(masked_systems, masked_labels)

        train_idx = np.zeros(len(dlabels), dtype=bool)
        train_idx[np.where(mask_nans)] = 1
        self.sl_kwargs.update({"train_idx": train_idx})
        train_idx_hist = self.sl_kwargs.get("train_idx_history")
        if train_idx_hist is None:
            train_idx_hist = []
        train_idx_hist.append(train_idx)
        self.sl_kwargs.update({"train_idx_history": train_idx_hist})

        preds, unc = self.predictor.predict(dsystems)

        # update predictions and store in history
        self.sl_kwargs.update({"predictions": preds})
        pred_hist = self.sl_kwargs.get("predictions_history")
        if pred_hist is None:
            pred_hist = []
        pred_hist.append(preds)
        self.sl_kwargs.update({"predictions_history": pred_hist})

        # update uncertainties and store in history
        self.sl_kwargs.update({"uncertainties": unc})
        unc_hist = self.sl_kwargs.get("uncertainties_history")
        if unc_hist is None:
            unc_hist = []
        unc_hist.append(unc)
        self.sl_kwargs.update({"uncertainties_history": unc_hist})

        # make sure haven't fully searched design space
        if any([np.isnan(label) for label in dlabels]):
            window = self.candidate_selector.target_window
            if not self.fixed_target:
                # update target window
                labels = self.design_space.design_space_labels
                if True not in np.isinf(window):
                    msg = "Movable bounds on target window not currently implemented"
                    raise SequentialLearnerError(msg)
                if not np.isinf(window[0]):
                    # maximization problem
                    # ie. window = (val, inf)
                    window[0] = np.maximum(np.nanmax(labels), window[0])
                elif not np.isinf(window[1]):
                    # minimization problem
                    # ie. window = (-inf, val)
                    window[1] = np.minimum(np.nanmin(labels), window[1])
                self.candidate_selector.target_window = window
            if window is not None:
                if self.target_window_history is None:
                    self.sl_kwargs.update({"target_window_history": [window]})
                else:
                    self.sl_kwargs["target_window_history"].append(window)
            cand_label_hist = None
            if self.candidate_selector.acquisition_function == "EIAbrupt":
                # get history of candidate labels for last three iterations
                cand_idx_hist = (
                    self.candidate_index_history if self.candidate_index_history else []
                )
                if len(cand_idx_hist) > 3:
                    # this is done in case less than three iterations have
                    # completed.
                    # even though EIAbrupt uses exactly last three iterations,
                    # so less than three will be discarded,
                    # this might still be useful to have for future AQF
                    # development?
                    cand_idx_hist = cand_idx_hist[-3:]

                if len(cand_idx_hist) == 0:
                    cand_label_hist = []
                else:
                    cand_label_hist = dlabels[[idx[0] for idx in cand_idx_hist]]
            # pick next candidate
            candidate_idx, _, aq_scores = self.candidate_selector.choose_candidate(
                design_space=self.design_space,
                allowed_idx=~train_idx,
                predictions=preds,
                uncertainties=unc,
                number_of_labelled_data_pts=sum(train_idx),
                cand_label_hist=cand_label_hist,
            )
        # if fully searched, no more candidate structures
        else:
            candidate_idx = None
            aq_scores = None
        aq_hist = self.sl_kwargs.get("acquisition_score_history")
        if aq_hist is None:
            aq_hist = []
        if aq_scores is not None:
            # new scores to add to history
            aq_hist.append(aq_scores)
            self.sl_kwargs.update({"acquisition_score_history": aq_hist})
        self.sl_kwargs.update({"candidate_indices": candidate_idx})
        self.sl_kwargs.update({"acquisition_scores": aq_scores})

        # update the candidate index history if new candidate
        if candidate_idx is not None:
            cand_idx_hist = self.sl_kwargs.get("candidate_index_history")
            if cand_idx_hist is None:
                cand_idx_hist = []
            cand_idx_hist.append(candidate_idx)
            self.sl_kwargs.update({"candidate_index_history": cand_idx_hist})

        # update the SL iteration count
        itc = self.sl_kwargs.get("iteration_count", 0)
        self.sl_kwargs.update({"iteration_count": itc + 1})

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        # get jsonified design space
        jsonified_ds = self.design_space.to_jsonified_dict()
        # get jsonified predictor
        jsonified_pred = self.predictor.to_jsonified_dict()
        # get jsonified candidate selector
        jsonified_cs = self.candidate_selector.to_jsonified_dict()
        # jsonify the sl kwargs
        jsonified_sl_kwargs = {}
        for k in self.sl_kwargs:
            if k != "iteration_count" and self.sl_kwargs[k] is not None:
                jsonified_sl_kwargs[k] = []
                for arr in self.sl_kwargs[k]:
                    if arr is not None:
                        jsonified_sl_kwargs[k].append(arr.tolist())
                    else:
                        jsonified_sl_kwargs[k].append(None)
            elif k == "iteration_count":
                jsonified_sl_kwargs["iteration_count"] = self.sl_kwargs[
                    "iteration_count"
                ]
            elif self.sl_kwargs[k] is None:
                jsonified_sl_kwargs[k] = None
        return {
            "design_space": jsonified_ds,
            "predictor": jsonified_pred,
            "candidate_selector": jsonified_cs,
            "sl_kwargs": jsonified_sl_kwargs,
        }

    def write_json_to_disk(self, write_location: str = ".", json_name: str = None):
        """
        Writes `SequentialLearner` to disk as a json
        """
        jsonified_list = self.to_jsonified_dict()

        if json_name is None:
            json_name = "sequential_learner.json"

        json_path = os.path.join(write_location, json_name)

        with open(json_path, "w") as f:
            json.dump(jsonified_list, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        if all_data.get("design_space") is None:
            msg = "DesignSpace must be provided"
            raise SequentialLearnerError(msg)
        design_space = DesignSpace.from_jsonified_dict(all_data["design_space"])
        predictor = Predictor.from_jsonified_dict(all_data.get("predictor", {}))
        candidate_selector = CandidateSelector.from_jsonified_dict(
            all_data.get("candidate_selector", {})
        )
        raw_sl_kwargs = all_data.get("sl_kwargs", {})
        sl_kwargs = {}
        for k in raw_sl_kwargs:
            if raw_sl_kwargs[k] is not None:
                if k in [
                    "predictions",
                    "uncertainties",
                    "acquisition_scores",
                    "candidate_indices",
                ]:
                    sl_kwargs[k] = np.array(raw_sl_kwargs[k])
                elif k in [
                    "predictions_history",
                    "uncertainties_history",
                    "candidate_index_history",
                    "acquisition_score_history",
                ]:
                    sl_kwargs[k] = [np.array(i) for i in raw_sl_kwargs[k]]
                elif k == "iteration_count":
                    sl_kwargs[k] = raw_sl_kwargs[k]
                elif k == "train_idx":
                    sl_kwargs[k] = np.array(raw_sl_kwargs[k], dtype=bool)
                elif k == "train_idx_history":
                    sl_kwargs[k] = [np.array(i, dtype=bool) for i in raw_sl_kwargs[k]]
                elif k == "target_window_history":
                    sl_kwargs[k] = [
                        [float(v[0]), float(v[1])] for v in raw_sl_kwargs[k]
                    ]
            else:
                sl_kwargs[k] = None

        return SequentialLearner(
            design_space=design_space,
            predictor=predictor,
            candidate_selector=candidate_selector,
            sl_kwargs=sl_kwargs,
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return SequentialLearner.from_jsonified_dict(all_data)


def multiple_simulated_sequential_learning_runs(
    full_design_space: DesignSpace,
    number_of_runs: int = 5,
    number_parallel_jobs: int = None,
    predictor: Predictor = None,
    candidate_selector: CandidateSelector = None,
    fixed_target: bool = True,
    init_training_size: int = 10,
    training_inclusion_window: Array = None,
    number_of_sl_loops: int = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    json_name_prefix: str = None,
) -> List[SequentialLearner]:
    """
    Conducts multiple simulated sequential learning runs

    Parameters
    ----------

    full_design_space:
        Fully labelled DesignSpace to simulate
        being searched over

    predictor:
        Predictor to be used for predicting properties while iterating.

    candidate_selector:
        CandidateSelector that specifies settings for candidate selection.
        This is where acquisition function, targets, etc. are
        specified.

    fixed_target:
        Whether to keep the target bounds fixed or move based on identified
        candidates. E.g. in a maximization problem keep target value fixed or
        change according to maximum observed value in search so far.
        Currently only implemented for maximization and minimization problems
        Defaults to having a fixed target

    init_training_size:
        Size of the initial training set to be selected from
        the full space.
        Default: 10

    training_inclusion_window:
        Window of target values that the initial training set can be selected from

    number_of_sl_loops:
        Integer specifying the number of sequential learning loops to be conducted.
        This value cannot be greater than
        `(DESIGN_SPACE_SIZE - init_training_size)/batch_size_to_add`
        Default: maximum number of sl loops calculated above

    number_of_runs:
        Integer of number of runs to be done
        Default: 5

    number_parallel_jobs:
        Integer giving the number of cores to be paralellized across
        using `joblib`
        Default: None (ie. will run in serial)

    write_to_disk:
        Boolean specifying whether runs history should be written to disk as jsons.
        Default: False

    write_location:
        String with the location where runs history jsons should be written to disk.
        Default: current directory

    json_name_prefix:
        Prefix used when writing out each simulated run as a json
        The naming convention is `{json_name_prefix}_{run #}.json`
        Default: acsl_run

    Returns
    -------

    runs_history:
        List of SequentialLearner objects for each simulated run
    """

    if number_parallel_jobs is not None:
        runs_history = Parallel(n_jobs=number_parallel_jobs)(
            delayed(simulated_sequential_learning)(
                full_design_space=full_design_space,
                predictor=predictor,
                candidate_selector=candidate_selector,
                number_of_sl_loops=number_of_sl_loops,
                init_training_size=init_training_size,
                fixed_target=fixed_target,
                training_inclusion_window=training_inclusion_window,
            )
            for i in range(number_of_runs)
        )

    else:
        runs_history = [
            simulated_sequential_learning(
                full_design_space=full_design_space,
                predictor=predictor,
                candidate_selector=candidate_selector,
                fixed_target=fixed_target,
                number_of_sl_loops=number_of_sl_loops,
                init_training_size=init_training_size,
                training_inclusion_window=training_inclusion_window,
            )
            for i in range(number_of_runs)
        ]

    # TODO: separate dictionary representation and writing to disk
    if write_to_disk:
        if not os.path.isdir(write_location):
            os.makedirs(write_location)
        if json_name_prefix is None:
            json_name_prefix = "acsl_run"
        for i, run in enumerate(runs_history):
            name = json_name_prefix + "_" + str(i) + ".json"
            run.write_json_to_disk(write_location=write_location, json_name=name)
        print(f"SL histories written to {write_location}")

    return runs_history


def simulated_sequential_learning(
    full_design_space: DesignSpace,
    predictor: Predictor = None,
    candidate_selector: CandidateSelector = None,
    fixed_target: bool = True,
    init_training_idx: Array = None,
    init_training_size: int = 10,
    training_inclusion_window: Array = None,
    number_of_sl_loops: int = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    json_name: str = None,
) -> SequentialLearner:
    """
    Conducts a simulated sequential learning loop for a
    fully labelled design space to explore.

    Parameters
    ----------

    full_design_space:
        Fully labelled DesignSpace to simulate
        being searched over

    predictor:
        Predictor to be used for predicting properties while iterating.

    candidate_selector:
        CandidateSelector that specifies settings for candidate selection.
        This is where acquisition function, targets, etc. are
        specified.

    fixed_target:
        Whether to keep the target bounds fixed or move based on identified
        candidates. E.g. in a maximization problem keep target value fixed or
        change according to maximum observed value in search so far.
        Currently only implemented for maximization and minimization problems
        Defaults to having a fixed target

    init_training_idx:
        Mask specifying initial training set to be used for the search. Must
        have the same size as the full design space provided
        Supercedes `init_training_size` and `training_inclusion_window`

    init_training_size:
        Size of the initial training set to be selected from
        the full space.
        Default: 10

    training_inclusion_window:
        Window of target values that the initial training set can be selected from

    number_of_sl_loops:
        Integer specifying the number of sequential learning loops to be conducted.
        This value cannot be greater than
        `(DESIGN_SPACE_SIZE - init_training_size)/batch_size_to_add`
        Default: maximum number of sl loops calculated above

    write_to_disk:
        Boolean specifying whether the resulting sequential learner should be
        written to disk as a json.
        Defaults to False.

    write_location:
        String with the location where the resulting sequential learner
        should be written to disk.
        Defaults to current directory.

    Returns
    -------

    sl:
        Sequential Learner after having been iterated as specified
        by the input settings. Contains candidate, prediction,
        and uncertainty histories for further analysis as desired.
    """

    ds_size = len(full_design_space)

    # check fully explored
    if True in np.isnan(full_design_space.design_space_labels):
        missing_label_idx = np.where(np.isnan(full_design_space.design_space_labels))[0]
        msg = (
            f"Design space must be fully explored."
            f" Missing labels at indices: {missing_label_idx}"
        )
        raise SequentialLearnerError(msg)

    if init_training_idx is not None:
        # initial training set provided
        if len(init_training_idx) != ds_size:
            msg = f"Initial training set mask size ({len(init_training_idx)})\
                  must match with design space size ({ds_size})"
            raise SequentialLearnerError(msg)
        init_training_size = sum(init_training_idx)
        init_idx = init_training_idx
    else:
        # need to generate initial training set
        # check that specified initial training size makes sense
        if init_training_size > ds_size:
            msg = f"Initial training size ({init_training_size})\
                 larger than design space ({ds_size})"
            raise SequentialLearnerError(msg)
        # generate initial training set
        init_idx = generate_initial_training_idx(
            training_set_size=init_training_size,
            design_space=full_design_space,
            inclusion_window=training_inclusion_window,
        )

    batch_size_to_add = candidate_selector.num_candidates_to_pick
    max_num_sl_loops = int(np.ceil((ds_size - init_training_size) / batch_size_to_add))

    if number_of_sl_loops is None:
        number_of_sl_loops = max_num_sl_loops

    # check that specified number of loops is feasible
    if number_of_sl_loops > max_num_sl_loops:
        msg = (
            f"Number of SL loops ({number_of_sl_loops}) cannot be greater than"
            f" ({max_num_sl_loops})"
        )
        raise SequentialLearnerError(msg)

    if full_design_space.feature_matrix is not None:
        init_systems = full_design_space.feature_matrix[np.where(init_idx)]
        use_feature_matrix = True
    else:
        init_systems = [
            full_design_space.design_space_structures[idx]
            for idx, b in enumerate(init_idx)
            if b
        ]
        use_feature_matrix = False
    init_labels = full_design_space.design_space_labels.copy()
    init_labels = init_labels[np.where(init_idx)]

    # default predictor settings
    if predictor is None:
        predictor = Predictor()

    # set up learner that is used for iteration
    dummy_labels = np.empty(len(full_design_space))
    dummy_labels[:] = np.nan
    ds = DesignSpace(
        design_space_structures=full_design_space.design_space_structures,
        design_space_labels=dummy_labels,
        feature_matrix=full_design_space.feature_matrix,
    )
    if use_feature_matrix:
        ds.update(feature_matrix=init_systems, labels=init_labels)
    else:
        ds.update(structures=init_systems, labels=init_labels)
    sl = SequentialLearner(
        design_space=ds,
        predictor=predictor,
        candidate_selector=candidate_selector,
        fixed_target=fixed_target,
    )
    # first iteration on initial dataset
    sl.iterate()

    # start simulated sequential learning loop
    for i in range(number_of_sl_loops):
        print(f"Sequential Learning Iteration #{i+1}")
        if sl.candidate_indices is not None:
            next_sys_indices = sl.candidate_indices
            next_labels = full_design_space.design_space_labels.take(next_sys_indices)
            if use_feature_matrix:
                next_systems = full_design_space.feature_matrix.take(
                    next_sys_indices, axis=0
                )
                sl.design_space.update(feature_matrix=next_systems, labels=next_labels)
            else:
                next_systems = sl.candidate_structures
                sl.design_space.update(structures=next_systems, labels=next_labels)
            sl.iterate()

    if write_to_disk:
        sl.write_json_to_disk(write_location=write_location, json_name=json_name)
        print(f"SL dictionary written to {write_location}")

    return sl


def generate_initial_training_idx(
    training_set_size: int,
    design_space: DesignSpace,
    inclusion_window: Array = None,
    rng: np.random.Generator = None,
):
    """
    Returns a mask for an initial training set to be used for a simulated search

    Parameters
    ----------

    training_set_size:
        Size of the initial training set to be selected

    design_space:
        DesignSpace to generate an initial training set for

    inclusion_window:
        When generating the initial training set, only include systems whose
        label falls within the specified window

    rng:
        Numpy random number generator (for testing)
    """
    if inclusion_window is None:
        inclusion_window = (-np.inf, np.inf)
    elif len(inclusion_window) != 2:
        msg = "Inclusion window must be defined by a tuple of exactly 2 bounds"
        raise Exception(msg)
    window = np.sort(inclusion_window)

    if rng is None:
        rng = np.random.default_rng()

    labels = design_space.design_space_labels
    possible_idx = np.where((labels > window[0]) & (labels < window[1]))[0]

    init_idx = np.zeros(len(design_space), dtype=bool)
    init_idx[rng.choice(possible_idx, size=training_set_size, replace=False)] = 1

    return init_idx


def get_overlap_score(mean: float, std: float, x2: float = None, x1: float = None):
    """Calculate overlap score given targets x2 (max) and x1 (min)"""
    if x1 is None and x2 is None:
        msg = "Please specify at least either a minimum or maximum target for MLI"
        raise SequentialLearnerError(msg)

    if x1 is None:
        x1 = -np.inf

    if x2 is None:
        x2 = np.inf

    norm_dist = stats.norm(loc=mean, scale=std)
    return norm_dist.cdf(x2) - norm_dist.cdf(x1)


def calculate_hhi_scores(
    structures: List[Atoms],
    hhi_type: str = "production",
    exclude_species: List[str] = None,
):
    """
    Calculates HHI scores for structures weighted by their composition.
    The scores are normalized and inverted such that these should
    be maximized in the interest of finding a low cost system

    Parameters
    ----------

    structures:
        List of Atoms objects for which to calculate the scores

    hhi_type:
        Type of HHI index to be used for the score
        Options
        - production (default)
        - reserves

    exclude_species:
        Species to be excluded when calculating the scores.
        An example use-case would be comparing transition-metal oxides
        where we can ignore the presence of O in each.

    Returns
    -------

    hhi_scores:
        Scores corresponding to each of the provided structures

    """
    if structures is None:
        msg = "To include HHI, the structures must be provided"
        raise SequentialLearnerError(msg)

    raw_hhi_data = HHI
    max_hhi = np.max([raw_hhi_data[hhi_type][r] for r in raw_hhi_data[hhi_type]])
    min_hhi = np.min([raw_hhi_data[hhi_type][r] for r in raw_hhi_data[hhi_type]])
    # normalize and invert (so that this score is to be maximized)
    norm_hhi_data = {
        el: 1.0 - (raw_hhi_data[hhi_type][el] - min_hhi) / (max_hhi - min_hhi)
        for el in raw_hhi_data[hhi_type]
    }

    hhi_scores = np.zeros(len(structures))
    for idx, struct in enumerate(structures):
        hhi = 0
        el_counts = struct.symbols.formula.count()
        if exclude_species is not None:
            for species in exclude_species:
                el_counts[species] = 0
        tot_size = sum(el_counts.values())
        # weight calculated hhi score by composition
        for el in el_counts:
            hhi += norm_hhi_data[el] * el_counts[el] / tot_size
        hhi_scores[idx] = hhi
    return hhi_scores


def calculate_segregation_energy_scores(
    structures: List[Atoms], data_source: str = "raban1999"
):
    """
    Calculates HHI scores for structures weighted by their composition.
    The scores are normalized and inverted such that these should
    be maximized in the interest of finding a low cost system

    Parameters
    ----------

    structures:
        List of Atoms objects for which to calculate the scores

    data_source:
        Which tabulated data should the segregation energies be pulled from.
        Options:
        - "raban1999": A.V. Raban, et. al. Phys. Rev. B 59, 15990
        - "rao2020": K. K. Rao, et. al. Topics in Catalysis volume 63, pages728-741 (2020)

    Returns
    -------

    hhi_scores:
        Scores corresponding to each of the provided structures

    """
    if structures is None:
        msg = "To include segregation energies, the structures must be provided"
        raise SequentialLearnerError(msg)

    if data_source == "raban1999":
        # won't consider surface energies (ie. dop == host) for normalization
        max_seg_ener = SEGREGATION_ENERGIES["raban1999"]["Pd"]["W"]
        min_seg_ener = SEGREGATION_ENERGIES["raban1999"]["Fe_100"]["Ag"]
        # normalize and invert (so that this score is to be maximized)
        norm_seg_ener_data = {}
        for hsp in SEGREGATION_ENERGIES["raban1999"]:
            norm_seg_ener_data[hsp] = {}
            for dsp in SEGREGATION_ENERGIES["raban1999"][hsp]:
                norm_seg_ener_data[hsp][dsp] = 1.0 - (
                    SEGREGATION_ENERGIES["raban1999"][hsp][dsp] - min_seg_ener
                ) / (max_seg_ener - min_seg_ener)
    elif data_source == "rao2020":
        norm_seg_ener_data = SEGREGATION_ENERGIES["rao2020"]
    else:
        msg = f"Unknown data source {data_source}"
        raise SequentialLearnerError(msg)

    seg_ener_scores = np.zeros(len(structures))
    for idx, struct in enumerate(structures):
        el_counts = struct.symbols.formula.count()
        assert len(el_counts) == 2
        for el in el_counts:
            if el_counts[el] == 1:
                dsp = el
            else:
                hsp = el
        seg_ener_scores[idx] = norm_seg_ener_data[hsp][dsp]
    return seg_ener_scores
