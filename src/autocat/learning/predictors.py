import copy
import numpy as np

from typing import List
from typing import Dict
from typing import Union
from prettytable import PrettyTable

from ase import Atoms

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from autocat.learning.featurizers import Featurizer
from autocat.learning.featurizers import (
    SUPPORTED_DSCRIBE_CLASSES,
    SUPPORTED_MATMINER_CLASSES,
)


class PredictorError(Exception):
    pass


class Predictor:
    def __init__(
        self,
        model_class=None,
        model_kwargs: Dict = None,  # TODO: kwargs -> options?
        featurizer_class=None,  # black
        featurization_kwargs: Dict = None,
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

        self._model_class = GaussianProcessRegressor
        self.model_class = model_class

        self._model_kwargs = None
        self.model_kwargs = model_kwargs

        self.regressor = self.model_class(
            **self.model_kwargs if self.model_kwargs else {}
        )

        self._featurizer_class = None
        self._featurization_kwargs = None

        self.featurizer_class = featurizer_class

        self.featurization_kwargs = featurization_kwargs

        self.featurizer = Featurizer(
            featurizer_class=self.featurizer_class,
            **self.featurization_kwargs if self.featurization_kwargs else {},
        )

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Predictor"]
        model_class_name = self.model_class.__module__ + "." + self.model_class.__name__
        pt.add_row(["class", model_class_name])
        pt.add_row(["kwargs", self.model_kwargs])
        pt.add_row(["is fit?", self.is_fit])
        feat_str = str(self.featurizer)
        return str(pt) + "\n" + feat_str

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
            self._model_kwargs = copy.deepcopy(model_kwargs)
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None
            self.regressor = self.model_class(**model_kwargs)

    @property
    def featurizer_class(self):
        return self._featurizer_class

    @featurizer_class.setter
    def featurizer_class(self, featurizer_class):
        if featurizer_class is not None:
            assert (
                featurizer_class in SUPPORTED_DSCRIBE_CLASSES
                or featurizer_class in SUPPORTED_MATMINER_CLASSES
            )
            self._featurizer_class = featurizer_class
            self._featurization_kwargs = None
            self.featurizer = Featurizer(featurizer_class,)
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None
            self.regressor = self.model_class(
                **self.model_kwargs if self.model_kwargs else {}
            )

    @property
    def featurization_kwargs(self):
        return self._featurization_kwargs

    @featurization_kwargs.setter
    def featurization_kwargs(self, featurization_kwargs):
        if featurization_kwargs is not None:
            assert isinstance(featurization_kwargs, dict)
            self._featurization_kwargs = featurization_kwargs.copy()
            self.featurizer = Featurizer(self.featurizer_class, **featurization_kwargs)
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None
            self.regressor = self.model_class(
                **self.model_kwargs if self.model_kwargs else {}
            )

    def copy(self):
        """
        Returns a copy
        """
        acp = self.__class__(
            model_class=self.model_class, featurizer_class=self.featurizer_class,
        )
        acp.regressor = copy.deepcopy(self.regressor)
        acp.is_fit = self.is_fit
        acp.featurization_kwargs = copy.deepcopy(self.featurization_kwargs)
        acp.model_kwargs = copy.deepcopy(self.model_kwargs)

        return acp

    def fit(
        self, training_structures: List[Union[Atoms, str]], y: np.ndarray,
    ):
        """
        Given a list of structures and labels will featurize
        and train a regression model

        Parameters
        ----------

        training_structures:
            List of structures to be trained upon

        y:
            Numpy array of labels corresponding to training structures
            of shape (# of training structures, # of targets)

        Returns
        -------

        trained_model:
            Trained `sklearn` model object
        """
        self.X_ = self.featurizer.featurize_multiple(training_structures)
        self.y_ = y
        self.regressor.fit(self.X_, self.y_)
        self.is_fit = True

    def predict(
        self, testing_structures: List[Atoms],
    ):
        """
        From a trained model, will predict on given structures

        Parameters
        ----------

        testing_structures:
            List of Atoms objects to make predictions on

        Returns
        -------

        predicted_labels:
            List of predicted labels for each input structure

        unc:
            List of uncertainties for each prediction if available.
            Otherwise returns `None`

        """
        assert self.is_fit
        featurized_input = self.featurizer.featurize_multiple(testing_structures)
        try:
            predicted_labels, unc = self.regressor.predict(
                featurized_input, return_std=True
            )
        except TypeError:
            predicted_labels = self.regressor.predict(featurized_input,)
            unc = None

        return predicted_labels, unc

    # TODO: "score" -> "get_scores"?
    def score(
        self,
        structures: List[Atoms],
        labels: np.ndarray,
        metric: str = "mae",
        return_predictions: bool = False,
        **kwargs,
    ):
        """
        Returns a prediction score given the actual corrections.

        Parameters
        ----------

        structures:
            List of Atoms objects of structures to be tested on

        labels:
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

        pred_label, unc = self.predict(structures)

        score_func = {"mae": mean_absolute_error, "mse": mean_squared_error}

        if metric not in score_func:
            msg = f"Metric: {metric} is not supported"
            raise PredictorError(msg)

        score = score_func[metric](labels, pred_label, **kwargs)

        if return_predictions:
            return score, pred_label, unc
        return score
