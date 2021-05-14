import numpy as np

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor

from autocat.learning.featurizers import get_X
from autocat.learning.featurizers import _get_number_of_features


class AutocatStructureCorrectorError(Exception):
    pass


class AutoCatStructureCorrector:
    def __init__(
        self,
        model_class=None,
        multiple_separate_models: bool = None,
        structure_featurizer: str = None,
        adsorbate_featurizer: str = None,
        maximum_structure_size: int = None,
        maximum_adsorbate_size: int = None,
        species_list: List[str] = None,
        refine_structures: bool = None,
        structure_featurization_kwargs: Dict = None,
        adsorbate_featurization_kwargs: Dict = None,
        model_kwargs: Dict = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        model_class:
            Class of regression model to be used for training and prediction.
            If this is changed after initialization, all previously set
            model_kwargs will be removed.
            N.B. must have fit and predict methods

        multiple_separate_models:
            Bool indicating whether to train separate models for each target output.
            If this is true, when fit to data, `acsc.regressor` will become the list
            of regressors with length of number of targets

        structure_featurizer:
            String giving featurizer to be used for full structure which will be
            fed into `autocat.learning.featurizers.full_structure_featurization`

        adsorbate_featurizer:
            String giving featurizer to be used for full structure which will be
            fed into `autocat.learning.featurizers.adsorbate_structure_featurization

        maximum_structure_size:
            Size of the largest structure to be supported by the representation.
            Default: number of atoms in largest structure within `structures`

        maximum_adsorbate_size:
            Integer giving the maximum adsorbate size to be encountered
            (ie. this determines if zero-padding should be applied and how much).
            If the provided value is less than the adsorbate size given by
            `adsorbate_indices`, representation will remain size of the adsorbate.
            Default: size of adsorbate provided

        species_list:
            List of species that could be encountered for featurization.
            Default: Parses over all `structures` and collects all encountered species

        refine_structures:
            Bool indicating whether the structures should be refined to include
            only the adsorbate and surface layer. Requires tags for all structures
            to have adsorbate atoms and surface atoms as 0 and 1, respectively

        """
        self.is_fit = False

        self._multiple_separate_models = False
        self.multiple_separate_models = multiple_separate_models

        self._model_class = GaussianProcessRegressor
        self.model_class = model_class

        self._model_kwargs = None
        self.model_kwargs = model_kwargs

        self.regressor = self.model_class(**self.model_kwargs or {})

        self._refine_structures = True
        self.refine_structures = refine_structures

        self._structure_featurizer = None
        self.structure_featurizer = structure_featurizer

        self._adsorbate_featurizer = None
        self.adsorbate_featurizer = adsorbate_featurizer

        self._structure_featurization_kwargs = None
        self.structure_featurization_kwargs = structure_featurization_kwargs

        self._adsorbate_featurization_kwargs = None
        self.adsorbate_featurization_kwargs = adsorbate_featurization_kwargs

        self._maximum_structure_size = None
        self.maximum_structure_size = maximum_structure_size

        self._maximum_adsorbate_size = None
        self.maximum_adsorbate_size = maximum_adsorbate_size

        self._species_list = None
        self.species_list = species_list

    @property
    def model_class(self):
        return self._model_class

    @model_class.setter
    def model_class(self, model_class):
        if model_class is not None:
            self._model_class = model_class
            # removes any model kwargs from previous model
            # if changed
            self._model_kwargs = None
            if self.is_fit:
                self.is_fit = False
            # generates new regressor with default settings
            self.regressor = self._model_class()

    @property
    def multiple_separate_models(self):
        return self._multiple_separate_models

    @multiple_separate_models.setter
    def multiple_separate_models(self, multiple_separate_models):
        if multiple_separate_models is not None:
            self._multiple_separate_models = multiple_separate_models
            if self.is_fit:
                self.is_fit = False

    @property
    def model_kwargs(self):
        return self._model_kwargs

    @model_kwargs.setter
    def model_kwargs(self, model_kwargs):
        if model_kwargs is not None:
            self._model_kwargs = model_kwargs
            if self.is_fit:
                self.is_fit = False
            self.regressor = self.model_class(**model_kwargs)

    @property
    def refine_structures(self):
        return self._refine_structures

    @refine_structures.setter
    def refine_structures(self, refine_structures):
        if refine_structures is not None:
            self._refine_structures = refine_structures
            if self.is_fit:
                self.is_fit = False

    @property
    def structure_featurizer(self):
        return self._structure_featurizer

    @structure_featurizer.setter
    def structure_featurizer(self, structure_featurizer):
        if structure_featurizer is not None:
            self._structure_featurizer = structure_featurizer
            if self.is_fit:
                self.is_fit = False

    @property
    def adsorbate_featurizer(self):
        return self._adsorbate_featurizer

    @adsorbate_featurizer.setter
    def adsorbate_featurizer(self, adsorbate_featurizer):
        if adsorbate_featurizer is not None:
            self._adsorbate_featurizer = adsorbate_featurizer
            if self.is_fit:
                self.is_fit = False

    @property
    def structure_featurization_kwargs(self):
        return self._structure_featurization_kwargs

    @structure_featurization_kwargs.setter
    def structure_featurization_kwargs(self, structure_featurization_kwargs):
        if structure_featurization_kwargs is not None:
            assert isinstance(structure_featurization_kwargs, dict)
            if self._structure_featurization_kwargs is not None:
                self._structure_featurization_kwargs = structure_featurization_kwargs
            if self.is_fit:
                self.is_fit = False

    @property
    def adsorbate_featurization_kwargs(self):
        return self._adsorbate_featurization_kwargs

    @adsorbate_featurization_kwargs.setter
    def adsorbate_featurization_kwargs(self, adsorbate_featurization_kwargs):
        if adsorbate_featurization_kwargs is not None:
            assert isinstance(adsorbate_featurization_kwargs, dict)
            self._adsorbate_featurization_kwargs = adsorbate_featurization_kwargs
            if self.is_fit:
                self.is_fit = False

    @property
    def maximum_structure_size(self):
        return self._maximum_structure_size

    @maximum_structure_size.setter
    def maximum_structure_size(self, maximum_structure_size):
        if maximum_structure_size is not None:
            self._maximum_structure_size = maximum_structure_size
            if self.is_fit:
                self.is_fit = False

    @property
    def maximum_adsorbate_size(self):
        return self._maximum_adsorbate_size

    @maximum_adsorbate_size.setter
    def maximum_adsorbate_size(self, maximum_adsorbate_size):
        if maximum_adsorbate_size is not None:
            self._maximum_adsorbate_size = maximum_adsorbate_size
            if self.is_fit:
                self.is_fit = False

    @property
    def species_list(self):
        return self._species_list

    @species_list.setter
    def species_list(self, species_list):
        if species_list is not None:
            self._species_list = species_list
            if self.is_fit:
                self.is_fit = False

    def get_total_number_of_features(self):
        # get specified kwargs for featurizers
        str_kwargs = self.structure_featurization_kwargs
        ads_kwargs = self.adsorbate_featurization_kwargs
        if str_kwargs is None:
            str_kwargs = {}
        if ads_kwargs is None:
            ads_kwargs = {}
        if self.structure_featurizer is not None:
            # check if one of dscribe structure featurizers
            if self.structure_featurizer in ["sine_matrix", "coulomb_matrix"]:
                str_kwargs.update({"n_atoms_max": self.maximum_structure_size})
            num_struct_feat = _get_number_of_features(
                self.structure_featurizer, **str_kwargs
            )
        else:
            # no structure featurizer present
            num_struct_feat = 0
        if self.adsorbate_featurizer is not None:
            # check if one of dscribe structure featurizers
            if self.adsorbate_featurizer == "soap":
                ads_kwargs.update({"species": self.species_list})
            num_ads_feat = _get_number_of_features(
                self.adsorbate_featurizer, **ads_kwargs
            )
        else:
            # no adsorbate featurizer present
            num_ads_feat = 0
        return num_struct_feat, num_ads_feat

    def fit(
        self,
        perturbed_structures: List[Union[Atoms, str]],
        corrections_list: List[np.ndarray] = None,
        correction_matrix: np.ndarray = None,
    ):
        """
        Given a list of perturbed structures
        will featurize and train a regression model on them

        Parameters
        ----------

        perturbed_structures:
            List of perturbed structures to be trained upon

        correction_matrix:
            Numpy array of collected matrices of perturbations corresponding to
            each of the perturbed structures.
            This can be generated via `autocat.perturbations.generate_perturbed_dataset`.
            Shape should be (# of structures, 3 * # of atoms in the largest structure)

        correction_list:
            List of np.arrays of correction vectors
            where each item is of shape (# of adsorbate atoms, 3).
            Adding the negative of these vectors to any perturbed
            structure should return it to the base structure

        Returns
        -------

        trained_model:
            Trained `sklearn` model object
        """
        X = get_X(
            perturbed_structures,
            maximum_structure_size=self.maximum_structure_size,
            structure_featurizer=self.structure_featurizer,
            maximum_adsorbate_size=self.maximum_adsorbate_size,
            adsorbate_featurizer=self.adsorbate_featurizer,
            species_list=self.species_list,
            refine_structures=self.refine_structures,
            structure_featurization_kwargs=self.structure_featurization_kwargs,
            adsorbate_featurization_kwargs=self.adsorbate_featurization_kwargs,
        )

        if self.maximum_structure_size is None:
            if self.refine_structures:
                ref_structures = [
                    structure[np.where(structure.get_tags() < 2)[0].tolist()]
                    for structure in perturbed_structures
                ]
                self.maximum_structure_size = max([len(ref) for ref in ref_structures])
            else:
                self.maximum_structure_size = max(
                    [len(s) for s in perturbed_structures]
                )

        if self.maximum_adsorbate_size is None:
            adsorbate_sizes = []
            for struct in perturbed_structures:
                adsorbate_sizes.append(
                    len(np.where(struct.get_tags() <= 0)[0].tolist())
                )
            self.maximum_adsorbate_size = max(adsorbate_sizes)

        if self.species_list is None:
            species_list = []
            for s in perturbed_structures:
                found_species = np.unique(s.get_chemical_symbols()).tolist()
                new_species = [
                    spec for spec in found_species if spec not in species_list
                ]
                species_list.extend(new_species)
            self.species_list = species_list

        if corrections_list is not None:
            correction_matrix = np.zeros(
                (len(corrections_list), 3 * self.maximum_adsorbate_size)
            )
            for idx, row in enumerate(corrections_list):
                correction_matrix[idx, : 3 * len(row)] = row.flatten()
        elif correction_matrix is not None:
            if correction_matrix.shape[1] != 3 * self.maximum_adsorbate_size:
                msg = f"Correction matrix must have {3 * self.maximum_adsorbate_size} targets, got {correction_matrix.shape[1]}"
                raise AutocatStructureCorrectorError(msg)
        else:
            msg = "Must specify either corrections list or matrix"
            raise AutocatStructureCorrectorError(msg)

        if not self.multiple_separate_models:
            self.regressor.fit(X, correction_matrix)
        else:
            regs = []
            for i in range(correction_matrix.shape[1]):
                reg = self.model_class(**self.model_kwargs or {})
                reg.fit(X, correction_matrix[:, i])
                regs.append(reg)
            assert regs[0] is not regs[1]
            self.regressor = regs

        self.is_fit = True

    def predict(
        self, initial_structure_guesses: List[Atoms],
    ):
        """
        From a trained model, will predict corrected structure
        of a given initial structure guess

        Parameters
        ----------

        initial_structure_guesses:
            List of Atoms objects of initial guesses for adsorbate
            placement to be optimized

        Returns
        -------

        predicted_corrections:
            List of corrections to be applied to each input structure

        corrected_structures:
            List of Atoms object with corrections applied

        unc:
            List of uncertainties for each prediction

        """
        assert self.is_fit
        featurized_input = get_X(
            structures=initial_structure_guesses,
            structure_featurizer=self.structure_featurizer,
            adsorbate_featurizer=self.adsorbate_featurizer,
            maximum_structure_size=self.maximum_structure_size,
            maximum_adsorbate_size=self.maximum_adsorbate_size,
            species_list=self.species_list,
            structure_featurization_kwargs=self.structure_featurization_kwargs,
            adsorbate_featurization_kwargs=self.adsorbate_featurization_kwargs,
        )
        if not self.multiple_separate_models:
            try:
                predicted_correction_matrix_full, unc = self.regressor.predict(
                    featurized_input, return_std=True
                )
            except TypeError:
                predicted_correction_matrix_full = self.regressor.predict(
                    featurized_input,
                )
                unc = None
        else:
            predicted_correction_matrix_full = np.zeros(
                (len(initial_structure_guesses), 3 * self.maximum_adsorbate_size)
            )
            try:
                uncs = np.zeros((len(initial_structure_guesses), len(self.regressor)))
                for idx, r in enumerate(self.regressor):
                    preds = r.predict(featurized_input, return_std=True)
                    predicted_correction_matrix_full[:, idx] = preds[0]
                    # uncertainties for target r for each struct to predict on
                    uncs[:, idx] = preds[1]
                # take average uncertainty for each struct
                unc = np.mean(uncs, axis=1)
            except TypeError:
                for idx, r in enumerate(self.regressor):
                    predicted_correction_matrix_full[:, idx] = r.predict(
                        featurized_input
                    )
                unc = None

        corrected_structures = [
            init_struct.copy() for init_struct in initial_structure_guesses
        ]

        corrected_structures = []
        predicted_corrections = []
        for idx, struct in enumerate(initial_structure_guesses):
            cs = struct.copy()
            list_of_adsorbate_indices = np.where(struct.get_tags() <= 0)[0].tolist()
            list_of_adsorbate_indices.sort()
            num_of_adsorbates = len(list_of_adsorbate_indices)
            corr = predicted_correction_matrix_full[
                idx, : 3 * num_of_adsorbates
            ].reshape(num_of_adsorbates, 3)
            predicted_corrections.append(corr)
            cs.positions[list_of_adsorbate_indices] -= corr
            corrected_structures.append(cs)

        return predicted_corrections, corrected_structures, unc

    def score(
        self,
        test_structure_guesses: List[Atoms],
        corrections_list: List[np.ndarray],
        metric: str = "mae",
        return_predictions: bool = False,
    ):
        """
        Returns a prediction score given the actual corrections.

        Parameters
        ----------

        test_structure_guesses:
            List of Atoms objects of structures to be tested o

        corrections_list:
            List of actual corrections as `np.arrays`

        metric:
            How the performance metric should be calculated
            Options:
            - mae: average of the average norm displacement vector difference
            - rmse: square root of the average of the average norm displacement
            vector difference squared

        return_predictions:
            Bool indicating whether the predictions and uncertainties should
            be returned in addition to the score

        Returns
        -------

        score:
            Float of calculated test score on the given data
        """
        assert self.is_fit

        pred_corr, _, unc = self.predict(test_structure_guesses)

        if metric == "mae":
            all_abs_vec_diff = []
            for i in range(len(pred_corr)):
                assert pred_corr[i].shape == corrections_list[i].shape
                N_i = len(pred_corr[i])
                abs_vec_diff = np.sum(
                    np.linalg.norm(corrections_list[i] - pred_corr[i], axis=1)
                )
                all_abs_vec_diff.append(abs_vec_diff / N_i)
            if return_predictions:
                return np.sum(all_abs_vec_diff) / len(all_abs_vec_diff), pred_corr, unc
            else:
                return np.sum(all_abs_vec_diff) / len(all_abs_vec_diff)
        elif metric == "rmse":
            all_sq_vec_diff = []
            for i in range(len(pred_corr)):
                assert pred_corr[i].shape == corrections_list[i].shape
                N_i = len(pred_corr[i])
                sq_vec_diff = np.sum(
                    np.linalg.norm(corrections_list[i] - pred_corr[i], axis=1) ** 2
                )
                all_sq_vec_diff.append(sq_vec_diff / N_i)
            if return_predictions:
                return (
                    np.sqrt(np.sum(all_sq_vec_diff) / len(all_sq_vec_diff)),
                    pred_corr,
                    unc,
                )
            else:
                return np.sqrt(np.sum(all_sq_vec_diff) / len(all_sq_vec_diff))
        else:
            msg = f"Metric: {metric} is not supported"
            raise AutocatStructureCorrectorError(msg)
