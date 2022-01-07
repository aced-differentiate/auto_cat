import copy
import numpy as np

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms
from sklearn import metrics

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from autocat.learning.featurizers import get_X
from autocat.learning.featurizers import _get_number_of_features


class PredictorError(Exception):
    pass


class Predictor:
    def __init__(
        self,
        model_class=None,
        multiple_separate_models: bool = None,
        structure_featurizer: str = None,
        adsorbate_featurizer: str = None,
        maximum_structure_size: int = None,
        maximum_adsorbate_size: int = None,
        elementalproperty_preset: str = None,
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

        self._elementalproperty_preset = "magpie"
        self.elementalproperty_preset = elementalproperty_preset

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
                self.X_ = None
                self.y_ = None
            # generates new regressor with default settings
            self.regressor = self._model_class()

    @property
    def model_kwargs(self):
        return self._model_kwargs

    @model_kwargs.setter
    def model_kwargs(self, model_kwargs):
        if model_kwargs is not None:
            self._model_kwargs = model_kwargs
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None
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
                self.X_ = None
                self.y_ = None

    @property
    def structure_featurizer(self):
        return self._structure_featurizer

    @structure_featurizer.setter
    def structure_featurizer(self, structure_featurizer):
        if structure_featurizer is not None:
            self._structure_featurizer = structure_featurizer
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

    @property
    def adsorbate_featurizer(self):
        return self._adsorbate_featurizer

    @adsorbate_featurizer.setter
    def adsorbate_featurizer(self, adsorbate_featurizer):
        if adsorbate_featurizer is not None:
            self._adsorbate_featurizer = adsorbate_featurizer
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

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
                self.X_ = None
                self.y_ = None

    @property
    def elementalproperty_preset(self):
        return self._elementalproperty_preset

    @elementalproperty_preset.setter
    def elementalproperty_preset(self, elementalproperty_preset):
        if elementalproperty_preset is not None:
            self._elementalproperty_preset = elementalproperty_preset
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

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
                self.X_ = None
                self.y_ = None

    @property
    def maximum_structure_size(self):
        return self._maximum_structure_size

    @maximum_structure_size.setter
    def maximum_structure_size(self, maximum_structure_size):
        if maximum_structure_size is not None:
            self._maximum_structure_size = maximum_structure_size
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

    @property
    def maximum_adsorbate_size(self):
        return self._maximum_adsorbate_size

    @maximum_adsorbate_size.setter
    def maximum_adsorbate_size(self, maximum_adsorbate_size):
        if maximum_adsorbate_size is not None:
            self._maximum_adsorbate_size = maximum_adsorbate_size
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

    @property
    def species_list(self):
        return self._species_list

    @species_list.setter
    def species_list(self, species_list):
        if species_list is not None:
            self._species_list = species_list
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

    def copy(self):
        """
        Returns a copy
        """
        acp = self.__class__(
            model_class=self.model_class,
            multiple_separate_models=self.multiple_separate_models,
            structure_featurizer=self.structure_featurizer,
            adsorbate_featurizer=self.adsorbate_featurizer,
            maximum_structure_size=self.maximum_structure_size,
            maximum_adsorbate_size=self.maximum_adsorbate_size,
            elementalproperty_preset=self.elementalproperty_preset,
            refine_structures=self.refine_structures,
        )
        acp.regressor = copy.deepcopy(self.regressor)
        acp.is_fit = self.is_fit
        acp.structure_featurization_kwargs = copy.deepcopy(
            self.structure_featurization_kwargs
        )
        acp.adsorbate_featurization_kwargs = copy.deepcopy(
            self.adsorbate_featurization_kwargs
        )
        acp.model_kwargs = copy.deepcopy(self.model_kwargs)
        if self.species_list is not None:
            acp.species_list = self.species_list.copy()

        return acp

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
        self, training_structures: List[Union[Atoms, str]], y: np.ndarray,
    ):
        """
        Given a list of perturbed structures
        will featurize and train a regression model on them

        Parameters
        ----------

        training_structures:
            List of perturbed structures to be trained upon

        y:
            Numpy array of labels corresponding to training structures
            of shape (# of training structures, # of targets)

        Returns
        -------

        trained_model:
            Trained `sklearn` model object
        """
        self.X_ = get_X(
            training_structures,
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
                    for structure in training_structures
                ]
                self.maximum_structure_size = max([len(ref) for ref in ref_structures])
            else:
                self.maximum_structure_size = max([len(s) for s in training_structures])

        if self.maximum_adsorbate_size is None:
            adsorbate_sizes = []
            for struct in training_structures:
                adsorbate_sizes.append(
                    len(np.where(struct.get_tags() <= 0)[0].tolist())
                )
            self.maximum_adsorbate_size = max(adsorbate_sizes)

        if self.species_list is None:
            species_list = []
            for s in training_structures:
                found_species = np.unique(s.get_chemical_symbols()).tolist()
                new_species = [
                    spec for spec in found_species if spec not in species_list
                ]
                species_list.extend(new_species)
            self.species_list = species_list
        self.y_ = y
        self.regressor.fit(self.X_, self.y_)
        self.is_fit = True

    def predict(
        self, testing_structures: List[Atoms],
    ):
        """
        From a trained model, will predict corrected structure
        of a given initial structure guess

        Parameters
        ----------

        testing_structures:
            List of Atoms objects of initial guesses for adsorbate
            placement to be optimized

        Returns
        -------

        predicted_corrections:
            List of corrections to be applied to each input structure

        unc:
            List of uncertainties for each prediction

        """
        assert self.is_fit
        featurized_input = get_X(
            structures=testing_structures,
            structure_featurizer=self.structure_featurizer,
            adsorbate_featurizer=self.adsorbate_featurizer,
            maximum_structure_size=self.maximum_structure_size,
            maximum_adsorbate_size=self.maximum_adsorbate_size,
            species_list=self.species_list,
            structure_featurization_kwargs=self.structure_featurization_kwargs,
            adsorbate_featurization_kwargs=self.adsorbate_featurization_kwargs,
        )
        try:
            predicted_labels, unc = self.regressor.predict(
                featurized_input, return_std=True
            )
        except TypeError:
            predicted_labels = self.regressor.predict(featurized_input,)
            unc = None

        return predicted_labels, unc

    def score(
        self,
        testing_structures: List[Atoms],
        y: np.ndarray,
        metric: str = "mae",
        return_predictions: bool = False,
        **kwargs,
    ):
        """
        Returns a prediction score given the actual corrections.

        Parameters
        ----------

        test_structure_guesses:
            List of Atoms objects of structures to be tested o

        y:
            Labels for the testing structures

        metric:
            How the performance metric should be calculated
            Options:
            - mae
            - mse

        return_predictions:
            Bool indicating whether the predictions and uncertainties should
            be returned in addition to the score

        Returns
        -------

        score:
            Float of calculated test score on the given data
        """
        assert self.is_fit

        pred_label, unc = self.predict(testing_structures)

        score_func = {"mae": mean_absolute_error, "mse": mean_squared_error}

        if metric not in score_func:
            msg = f"Metric: {metric} is not supported"
            raise PredictorError(msg)

        score = score_func[metric](y, pred_label, **kwargs)

        if return_predictions:
            return score, pred_label, unc
        return score
