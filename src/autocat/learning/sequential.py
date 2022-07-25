import copy
import os
import json
from typing import List
from typing import Dict

import numpy as np
from joblib import Parallel, delayed
from prettytable import PrettyTable
from ase import Atoms
from ase.io.jsonio import encode as atoms_encoder
from ase.io.jsonio import decode as atoms_decoder
from scipy import stats

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

    def to_jsonified_dict(self) -> Dict:
        """
        Returns a jsonified dict representation
        """
        collected_structs = []
        for struct in self.design_space_structures:
            collected_structs.append(atoms_encoder(struct))
        jsonified_labels = [float(x) for x in self.design_space_labels]
        return {"structures": collected_structs, "labels": jsonified_labels}

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
        if all_data.get("structures") is None or all_data.get("labels") is None:
            msg = "Both structures and labels must be provided"
            raise DesignSpaceError(msg)
        try:
            structures = []
            for encoded_atoms in all_data["structures"]:
                structures.append(atoms_decoder(encoded_atoms))
        except (json.JSONDecodeError, TypeError):
            msg = "Please ensure design space structures encoded using `ase.io.jsonio.encode`"
            raise DesignSpaceError(msg)
        labels = np.array(all_data["labels"])
        return DesignSpace(
            design_space_structures=structures, design_space_labels=labels,
        )

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return DesignSpace.from_jsonified_dict(all_data)


class CandidateSelectorError(Exception):
    pass


class CandidateSelector:
    def __init__(
        self,
        acquisition_function: str = None,
        num_candidates_to_pick: int = None,
        target_window: Array = None,
        include_hhi: bool = None,
        hhi_type: str = "production",
        include_segregation_energies: bool = None,
        segregation_energy_data_source: str = None,
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
        """
        self._acquisition_function = "Random"
        self.acquisition_function = acquisition_function

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

    @property
    def acquisition_function(self):
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function):
        if acquisition_function is not None:
            if acquisition_function in ["MLI", "MU", "Random"]:
                self._acquisition_function = acquisition_function
            else:
                msg = f"Unrecognized acquisition function {acquisition_function}\
                     Please select one of 'MLI', 'MU', or 'Random'"
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
                msg = f"Unrecognized segregation energy data source {segregation_energy_data_source}.\
                     Please select one of 'raban1999' or 'rao2020'"
                raise CandidateSelectorError(msg)

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Candidate Selector"]
        pt.add_row(["acquisition function", self.acquisition_function])
        pt.add_row(["# of candidates to pick", self.num_candidates_to_pick])
        pt.add_row(["target window", self.target_window])
        pt.add_row(["include hhi?", self.include_hhi])
        if self.include_hhi:
            pt.add_row(["hhi type", self.hhi_type])
        pt.add_row(["include segregation energies?", self.include_segregation_energies])
        pt.add_row(
            ["segregation energies data source", self.segregation_energy_data_source]
        )
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
            return np.array_equal(self.target_window, other.target_window)
        return False

    def copy(self):
        """
        Returns a copy of the CandidateSelector
        """
        cs = self.__class__(
            acquisition_function=self.acquisition_function,
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

        if allowed_idx is None:
            if True in np.isnan(design_space.design_space_labels):
                allowed_idx = np.where(np.isnan(design_space.design_space_labels))[0]
            else:
                allowed_idx = np.ones(ds_size, dtype=bool)

        hhi_scores = np.ones(ds_size)
        if self.include_hhi:
            hhi_scores = calculate_hhi_scores(
                design_space.design_space_structures, self.hhi_type
            )

        segreg_energy_scores = np.ones(ds_size)
        if self.include_segregation_energies:
            segreg_energy_scores = calculate_segregation_energy_scores(
                design_space.design_space_structures
            )

        aq = self.acquisition_function
        if aq == "Random":
            aq_scores = (
                np.random.choice(ds_size, size=ds_size, replace=False)
                * hhi_scores
                * segreg_energy_scores
            )

        elif aq == "MU":
            if uncertainties is None:
                msg = "For 'MU', the uncertainties must be supplied"
                raise CandidateSelectorError(msg)
            aq_scores = uncertainties.copy() * hhi_scores * segreg_energy_scores

        elif aq == "MLI":
            if uncertainties is None or predictions is None:
                msg = "For 'MLI', both uncertainties and predictions must be supplied"
                raise CandidateSelectorError(msg)
            target_window = self.target_window
            aq_scores = (
                np.array(
                    [
                        get_overlap_score(
                            mean, std, x2=target_window[1], x1=target_window[0]
                        )
                        for mean, std in zip(predictions, uncertainties)
                    ]
                )
                * hhi_scores
                * segreg_energy_scores
            )

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
            "num_candidates_to_pick": self.num_candidates_to_pick,
            "target_window": target_window,
            "include_hhi": self.include_hhi,
            "hhi_type": self.hhi_type,
            "include_segregation_energies": self.include_segregation_energies,
            "segregation_energy_data_source": self.segregation_energy_data_source,
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
        return CandidateSelector(
            acquisition_function=all_data.get("acquisition_function"),
            num_candidates_to_pick=all_data.get("num_candidates_to_pick"),
            target_window=target_window,
            include_hhi=all_data.get("include_hhi"),
            hhi_type=all_data.get("hhi_type"),
            include_segregation_energies=all_data.get("include_segregation_energies"),
            segregation_energy_data_source=all_data.get(
                "segregation_energy_data_source"
            ),
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
        """
        # TODO: move predefined attributes (train_idx, candidate_idxs) to a
        # different container (not kwargs)

        self._design_space = None
        self.design_space = design_space.copy()

        self._predictor = Predictor()
        self.predictor = predictor

        self._candidate_selector = CandidateSelector()
        self.candidate_selector = candidate_selector

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
            candidate_idx, _, aq_scores = self.candidate_selector.choose_candidate(
                design_space=self.design_space,
                allowed_idx=~train_idx,
                predictions=preds,
                uncertainties=unc,
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

    predictor:
        Predictor to be used for predicting properties while iterating.

    candidate_selector:
        CandidateSelector that specifies settings for candidate selection.
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
                predictor=predictor,
                candidate_selector=candidate_selector,
                number_of_sl_loops=number_of_sl_loops,
                init_training_size=init_training_size,
            )
            for i in range(number_of_runs)
        )

    else:
        runs_history = [
            simulated_sequential_learning(
                full_design_space=full_design_space,
                predictor=predictor,
                candidate_selector=candidate_selector,
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
    predictor: Predictor = None,
    candidate_selector: CandidateSelector = None,
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

    predictor:
        Predictor to be used for predicting properties while iterating.

    candidate_selector:
        CandidateSelector that specifies settings for candidate selection.
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

    # default predictor settings
    if predictor is None:
        predictor = Predictor()

    # set up learner that is used for iteration
    dummy_labels = np.empty(len(full_design_space))
    dummy_labels[:] = np.nan
    ds = DesignSpace(full_design_space.design_space_structures, dummy_labels)
    ds.update(init_structs, init_labels)
    sl = SequentialLearner(
        design_space=ds, predictor=predictor, candidate_selector=candidate_selector,
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
