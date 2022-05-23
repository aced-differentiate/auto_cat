import copy
import os
import json
import importlib
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
from sklearn.gaussian_process import GaussianProcessRegressor
from dscribe.descriptors import SineMatrix

from autocat.learning.predictors import Predictor
from autocat.data.hhi import HHI
from autocat.data.segregation_energies import SEGREGATION_ENERGIES


Array = List[float]


class DesignSpaceError(Exception):
    pass


class DesignSpace:
    def __init__(
        self, design_space_structures: List[Atoms], design_space_labels: Array,
    ):
        """
        Constructor.

        Parameters
        ----------

        design_space_structures:
            List of all structures within the design space

        design_space_labels:
            Labels corresponding to all structures within the design space.
            If label not yet known, set to np.nan

        """
        if len(design_space_structures) != design_space_labels.shape[0]:
            msg = f"Number of structures ({len(design_space_structures)})\
                 and labels ({design_space_labels.shape[0]}) must match"
            raise DesignSpaceError(msg)

        self._design_space_structures = [
            struct.copy() for struct in design_space_structures
        ]
        self._design_space_labels = design_space_labels.copy()

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "DesignSpace"]
        pt.add_row(["total # of systems", len(self)])
        num_unknown = sum(np.isnan(self.design_space_labels))
        pt.add_row(["# of unlabelled systems", num_unknown])
        pt.add_row(["unique species present", self.species_list])
        max_label = max(self.design_space_labels)
        pt.add_row(["maximum label", max_label])
        min_label = min(self.design_space_labels)
        pt.add_row(["minimum label", min_label])
        pt.max_width = 70
        return str(pt)

    def __len__(self):
        return len(self.design_space_structures)

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
    def species_list(self):
        species_list = []
        for s in self.design_space_structures:
            # get all unique species
            found_species = np.unique(s.get_chemical_symbols()).tolist()
            new_species = [spec for spec in found_species if spec not in species_list]
            species_list.extend(new_species)
        return species_list

    def update(self, structures: List[Atoms], labels: Array):
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
        if (structures is not None) and (labels is not None):
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

    def to_jsonified_list(self) -> List:
        """
        Returns a jsonified list representation
        """
        collected_jsons = []
        for struct in self.design_space_structures:
            collected_jsons.append(atoms_encoder(struct))
        # append labels to list of collected jsons
        jsonified_labels = [float(x) for x in self.design_space_labels]
        collected_jsons.append(jsonified_labels)
        return collected_jsons

    def write_json_to_disk(
        self,
        json_name: str = None,
        write_location: str = ".",
        write_to_disk: bool = True,
    ):
        """
        Writes DesignSpace to disk as a json
        """
        collected_jsons = self.to_jsonified_list()
        # set default json name if needed
        if json_name is None:
            json_name = "acds.json"
        # write out single json
        if write_to_disk:
            json_path = os.path.join(write_location, json_name)
            with open(json_path, "w") as f:
                json.dump(collected_jsons, f)

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        structures = []
        for i in range(len(all_data) - 1):
            atoms = atoms_decoder(all_data[i])
            structures.append(atoms)
        labels = np.array(all_data[-1])
        return DesignSpace(
            design_space_structures=structures, design_space_labels=labels,
        )


class SequentialLearnerError(Exception):
    pass


# TODO: "kwargs" -> "options"?
class SequentialLearner:
    def __init__(
        self,
        design_space: DesignSpace,
        predictor_kwargs: Dict[str, Union[str, float]] = None,
        candidate_selection_kwargs: Dict[str, Union[str, float]] = None,
        sl_kwargs: Dict[str, int] = None,
    ):
        # TODO: move predefined attributes (train_idx, candidate_idxs) to a
        # different container (not kwargs)

        self._design_space = None
        self.design_space = design_space.copy()

        # predictor arguments to use throughout the SL process
        if predictor_kwargs is None:
            predictor_kwargs = {
                "model_class": GaussianProcessRegressor,
                "featurizer_class": SineMatrix,
            }
        if "model_class" not in predictor_kwargs:
            predictor_kwargs["model_class"] = GaussianProcessRegressor
        if "featurizer_class" not in predictor_kwargs:
            predictor_kwargs["featurizer_class"] = SineMatrix
        if "featurization_kwargs" not in predictor_kwargs:
            predictor_kwargs["featurization_kwargs"] = {}
        ds_structs_kwargs = {
            "design_space_structures": design_space.design_space_structures
        }
        predictor_kwargs["featurization_kwargs"].update(ds_structs_kwargs)
        self._predictor_kwargs = None
        self.predictor_kwargs = predictor_kwargs
        self._predictor = Predictor(**predictor_kwargs)

        # acquisition function arguments to use for candidate selection
        if not candidate_selection_kwargs:
            candidate_selection_kwargs = {"aq": "Random"}
        self._candidate_selection_kwargs = None
        self.candidate_selection_kwargs = candidate_selection_kwargs

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
        pt.add_row(["acquisition function", self.candidate_selection_kwargs.get("aq")])
        pt.add_row(
            [
                "# of candidates to pick",
                self.candidate_selection_kwargs.get("num_candidates_to_pick", 1),
            ]
        )
        pt.add_row(
            ["target maximum", self.candidate_selection_kwargs.get("target_max")]
        )
        pt.add_row(
            ["target minimum", self.candidate_selection_kwargs.get("target_min")]
        )
        pt.add_row(
            ["include hhi?", self.candidate_selection_kwargs.get("include_hhi", False)]
        )
        pt.add_row(
            [
                "include segregation energies?",
                self.candidate_selection_kwargs.get("include_seg_ener", False),
            ]
        )
        return str(pt) + "\n" + str(self.design_space) + "\n" + str(self.predictor)

    @property
    def design_space(self):
        return self._design_space

    @design_space.setter
    def design_space(self, design_space):
        self._design_space = design_space

    @property
    def predictor_kwargs(self):
        return self._predictor_kwargs

    @predictor_kwargs.setter
    def predictor_kwargs(self, predictor_kwargs):
        if predictor_kwargs is None:
            predictor_kwargs = {
                "model_class": GaussianProcessRegressor,
                "featurizer_class": SineMatrix,
            }
        if "model_class" not in predictor_kwargs:
            predictor_kwargs["model_class"] = GaussianProcessRegressor
        if "featurizer_class" not in predictor_kwargs:
            predictor_kwargs["featurizer_class"] = SineMatrix
        if "featurization_kwargs" not in predictor_kwargs:
            predictor_kwargs["featurization_kwargs"] = {}
        ds_structs_kwargs = {
            "design_space_structures": self.design_space.design_space_structures
        }
        predictor_kwargs["featurization_kwargs"].update(ds_structs_kwargs)
        self._predictor_kwargs = copy.deepcopy(predictor_kwargs)
        self._predictor = Predictor(**predictor_kwargs)

    @property
    def predictor(self):
        return self._predictor

    @property
    def candidate_selection_kwargs(self):
        return self._candidate_selection_kwargs

    @candidate_selection_kwargs.setter
    def candidate_selection_kwargs(self, candidate_selection_kwargs):
        if not candidate_selection_kwargs:
            candidate_selection_kwargs = {}
        self._candidate_selection_kwargs = candidate_selection_kwargs.copy()

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
    def candidate_structures(self):
        idxs = self.candidate_indices
        if idxs is not None:
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

    def copy(self):
        """
        Returns a copy
        """
        acsl = self.__class__(design_space=self.design_space,)
        acsl.predictor_kwargs = copy.deepcopy(self.predictor_kwargs)
        acsl.sl_kwargs = copy.deepcopy(self.sl_kwargs)
        return acsl

    def iterate(self):
        """Runs the next iteration of sequential learning.

        This process consists of:
        - retraining the predictor
        - predicting candidate properties and calculating candidate scores (if
        fully explored returns None)
        - selecting the next batch of candidates for objective evaluation (if
        fully explored returns None)
        """

        dstructs = self.design_space.design_space_structures
        dlabels = self.design_space.design_space_labels

        mask_nans = ~np.isnan(dlabels)
        masked_structs = [struct for i, struct in enumerate(dstructs) if mask_nans[i]]
        masked_labels = dlabels[np.where(mask_nans)]

        self.predictor.fit(masked_structs, masked_labels)

        train_idx = np.zeros(len(dlabels), dtype=bool)
        train_idx[np.where(mask_nans)] = 1
        self.sl_kwargs.update({"train_idx": train_idx})
        train_idx_hist = self.sl_kwargs.get("train_idx_history")
        if train_idx_hist is None:
            train_idx_hist = []
        train_idx_hist.append(train_idx)
        self.sl_kwargs.update({"train_idx_history": train_idx_hist})

        preds, unc = self.predictor.predict(dstructs)

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
            candidate_idx, _, aq_scores = choose_next_candidate(
                dstructs,
                dlabels,
                train_idx,
                preds,
                unc,
                **self.candidate_selection_kwargs,
            )
        # if fully searched, no more candidate structures
        else:
            candidate_idx = None
            aq_scores = None
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

    def to_jsonified_list(self) -> List:
        """
        Returns a jsonified list representation
        """
        jsonified_list = self.design_space.to_jsonified_list()
        # append kwargs for predictor
        jsonified_pred_kwargs = {}
        for k in self.predictor_kwargs:
            if k in ["model_class", "featurizer_class"]:
                mod_string = self.predictor_kwargs[k].__module__
                class_string = self.predictor_kwargs[k].__name__
                jsonified_pred_kwargs[k] = [mod_string, class_string]
            elif k == "featurization_kwargs":
                jsonified_pred_kwargs[k] = copy.deepcopy(self.predictor_kwargs[k])
                # assumes design space will always match DesignSpace
                del jsonified_pred_kwargs[k]["design_space_structures"]
            else:
                jsonified_pred_kwargs[k] = self.predictor_kwargs[k]
        jsonified_list.append(jsonified_pred_kwargs)
        # append kwargs for candidate selection
        jsonified_list.append(self.candidate_selection_kwargs)
        # append the acsl kwargs
        jsonified_sl_kwargs = {}
        for k in self.sl_kwargs:
            if k != "iteration_count" and self.sl_kwargs[k] is not None:
                jsonified_sl_kwargs[k] = [arr.tolist() for arr in self.sl_kwargs[k]]
            elif k == "iteration_count":
                jsonified_sl_kwargs["iteration_count"] = self.sl_kwargs[
                    "iteration_count"
                ]
            elif self.sl_kwargs[k] is None:
                jsonified_sl_kwargs[k] = None
        jsonified_list.append(jsonified_sl_kwargs)
        return jsonified_list

    def write_json_to_disk(self, write_location: str = ".", json_name: str = None):
        """
        Writes `SequentialLearner` to disk as a json
        """
        jsonified_list = self.to_jsonified_list()

        if json_name is None:
            json_name = "acsl.json"

        json_path = os.path.join(write_location, json_name)

        with open(json_path, "w") as f:
            json.dump(jsonified_list, f)

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        structures = []
        for i in range(len(all_data) - 4):
            atoms = atoms_decoder(all_data[i])
            structures.append(atoms)
        labels = np.array(all_data[-4])
        acds = DesignSpace(
            design_space_structures=structures, design_space_labels=labels,
        )
        predictor_kwargs = all_data[-3]
        for k in predictor_kwargs:
            if k in ["model_class", "featurizer_class"]:
                mod = importlib.import_module(predictor_kwargs[k][0])
                predictor_kwargs[k] = getattr(mod, predictor_kwargs[k][1])
        candidate_selection_kwargs = all_data[-2]
        raw_sl_kwargs = all_data[-1]
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
                ]:
                    sl_kwargs[k] = [np.array(i) for i in raw_sl_kwargs[k]]
                elif k == "iteration_count":
                    sl_kwargs[k] = raw_sl_kwargs[k]
                elif k == "train_idx":
                    sl_kwargs[k] = np.array(raw_sl_kwargs[k], dtype=bool)
                elif k == "train_idx_history":
                    sl_kwargs[k] = [np.array(i, dtype=bool) for i in raw_sl_kwargs[k]]
            else:
                sl_kwargs[k] = None

        return SequentialLearner(
            design_space=acds,
            predictor_kwargs=predictor_kwargs,
            candidate_selection_kwargs=candidate_selection_kwargs,
            sl_kwargs=sl_kwargs,
        )


def multiple_simulated_sequential_learning_runs(
    full_design_space: DesignSpace,
    number_of_runs: int = 5,
    number_parallel_jobs: int = None,
    predictor_kwargs: Dict[str, Union[str, float]] = None,
    candidate_selection_kwargs: Dict[str, Union[str, float]] = None,
    init_training_size: int = 10,
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

    predictor_kwargs:
        Kwargs to be used in setting up the predictor.
        This is where model class, model hyperparameters, etc.
        are specified.

    candidate_selection_kwargs:
        Kwargs that specify that settings for candidate selection.
        This is where acquisition function, targets, etc. are
        specified.

    init_training_size:
        Size of the initial training set to be selected from
        the full space.
        Default: 10

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
                predictor_kwargs=predictor_kwargs,
                candidate_selection_kwargs=candidate_selection_kwargs,
                number_of_sl_loops=number_of_sl_loops,
                init_training_size=init_training_size,
            )
            for i in range(number_of_runs)
        )

    else:
        runs_history = [
            simulated_sequential_learning(
                full_design_space=full_design_space,
                predictor_kwargs=predictor_kwargs,
                candidate_selection_kwargs=candidate_selection_kwargs,
                number_of_sl_loops=number_of_sl_loops,
                init_training_size=init_training_size,
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
    predictor_kwargs: Dict[str, Union[str, float]] = None,
    candidate_selection_kwargs: Dict[str, Union[str, float]] = None,
    init_training_size: int = 10,
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

    predictor_kwargs:
        Kwargs to be used in setting up the predictor.
        This is where model class, model hyperparameters, etc.
        are specified.

    candidate_selection_kwargs:
        Kwargs that specify that settings for candidate selection.
        This is where acquisition function, targets, etc. are
        specified.

    init_training_size:
        Size of the initial training set to be selected from
        the full space.
        Default: 10

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

    # check that specified initial training size makes sense
    if init_training_size > ds_size:
        msg = f"Initial training size ({init_training_size})\
             larger than design space ({ds_size})"
        raise SequentialLearnerError(msg)

    batch_size_to_add = candidate_selection_kwargs.get("num_candidates_to_pick", 1)
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

    # generate initial training set
    init_idx = np.zeros(ds_size, dtype=bool)
    init_idx[np.random.choice(ds_size, init_training_size, replace=False)] = 1

    init_structs = [
        full_design_space.design_space_structures[idx]
        for idx, b in enumerate(init_idx)
        if b
    ]
    init_labels = full_design_space.design_space_labels.copy()
    init_labels = init_labels[np.where(init_idx)]

    # set up learner that is used for iteration
    dummy_labels = np.empty(len(full_design_space))
    dummy_labels[:] = np.nan
    ds = DesignSpace(full_design_space.design_space_structures, dummy_labels)
    ds.update(init_structs, init_labels)
    sl = SequentialLearner(
        design_space=ds,
        predictor_kwargs=predictor_kwargs,
        candidate_selection_kwargs=candidate_selection_kwargs,
    )
    # first iteration on initial dataset
    sl.iterate()

    # start simulated sequential learning loop
    for i in range(number_of_sl_loops):
        print(f"Sequential Learning Iteration #{i+1}")
        if sl.candidate_indices is not None:
            next_structs = sl.candidate_structures
            next_labels = full_design_space.design_space_labels.take(
                sl.candidate_indices
            )
            sl.design_space.update(next_structs, next_labels)
            sl.iterate()

    if write_to_disk:
        sl.write_json_to_disk(write_location=write_location, json_name=json_name)
        print(f"SL dictionary written to {write_location}")

    return sl


def choose_next_candidate(
    structures: List[Atoms] = None,
    labels: Array = None,
    train_idx: Array = None,
    pred: Array = None,
    unc: Array = None,
    aq: str = "MLI",
    num_candidates_to_pick: int = None,
    target_min: float = None,
    target_max: float = None,
    include_hhi: bool = False,
    hhi_type: str = "production",
    include_seg_ener: bool = False,
):
    """
    Chooses the next candidate(s) from a given acquisition function

    Parameters
    ----------

    structures:
        List of `Atoms` objects to be used for HHI weighting if desired

    labels:
        Array of the labels for the data

    train_idx:
        Indices of all data entries already in the training set
        Default: consider entire training set

    pred:
        Predictions for all structures in the dataset

    unc:
        Uncertainties for all structures in the dataset

    aq:
        Acquisition function to be used to select the next candidates
        Options
        - MLI: maximum likelihood of improvement (default)
        - Random
        - MU: maximum uncertainty

    num_candidates_to_pick:
        Number of candidates to choose from the dataset

    target_min:
        Minimum target value to optimize for

    target_max:
        Maximum target value to optimize for

    include_hhi:
        Whether HHI scores should be used to weight aq scores

    hhi_type:
        Type of HHI index to be used for weighting
        Options
        - production (default)
        - reserves

    include_seg_ener:
        Whether segregation energies should be used to weight aq scores

    Returns
    -------

    parent_idx:
        Index/indices of the selected candidates

    max_scores:
        Maximum scores (corresponding to the selected candidates for given `aq`)

    aq_scores:
        Calculated scores based on the selected `aq` for the entire training set
    """
    hhi_scores = None
    if include_hhi:
        if structures is None:
            msg = "Structures must be provided to include HHI scores"
            raise SequentialLearnerError(msg)
        hhi_scores = calculate_hhi_scores(structures, hhi_type)

    segreg_energy_scores = None
    if include_seg_ener:
        if structures is None:
            msg = "Structures must be provided to include segregation energy scores"
            raise SequentialLearnerError(msg)
        segreg_energy_scores = calculate_segregation_energy_scores(structures)

    if aq == "Random":
        if labels is None:
            msg = "For aq = 'Random', the labels must be supplied"
            raise SequentialLearnerError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(labels), dtype=bool)

        if hhi_scores is None:
            hhi_scores = np.ones(len(train_idx))

        if segreg_energy_scores is None:
            segreg_energy_scores = np.ones(len(train_idx))

        aq_scores = (
            np.random.choice(len(labels), size=len(labels), replace=False)
            * hhi_scores
            * segreg_energy_scores
        )

    elif aq == "MU":
        if unc is None:
            msg = "For aq = 'MU', the uncertainties must be supplied"
            raise SequentialLearnerError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(unc), dtype=bool)

        if hhi_scores is None:
            hhi_scores = np.ones(len(train_idx))

        if segreg_energy_scores is None:
            segreg_energy_scores = np.ones(len(train_idx))

        aq_scores = unc.copy() * hhi_scores * segreg_energy_scores

    elif aq == "MLI":
        if unc is None or pred is None:
            msg = "For aq = 'MLI', both uncertainties and predictions must be supplied"
            raise SequentialLearnerError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(unc), dtype=bool)

        if hhi_scores is None:
            hhi_scores = np.ones(len(train_idx))

        if segreg_energy_scores is None:
            segreg_energy_scores = np.ones(len(train_idx))

        aq_scores = (
            np.array(
                [
                    get_overlap_score(mean, std, x2=target_max, x1=target_min)
                    for mean, std in zip(pred, unc)
                ]
            )
            * hhi_scores
            * segreg_energy_scores
        )

    else:
        msg = f"Acquisition function {aq} is not supported"
        raise NotImplementedError(msg)

    if num_candidates_to_pick is None:
        next_idx = np.array([np.argmax(aq_scores[~train_idx])])
        max_scores = [np.max(aq_scores[~train_idx])]

    else:
        next_idx = np.argsort(aq_scores[~train_idx])[-num_candidates_to_pick:]
        sorted_array = aq_scores[~train_idx][next_idx]
        max_scores = list(sorted_array[-num_candidates_to_pick:])
    parent_idx = np.arange(aq_scores.shape[0])[~train_idx][next_idx]

    return parent_idx, max_scores, aq_scores


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


def calculate_hhi_scores(structures: List[Atoms], hhi_type: str = "production"):
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
        tot_size = len(struct)
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
