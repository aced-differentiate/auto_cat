import copy
import os
import importlib
import json
import numpy as np

from typing import List
from typing import Dict
from typing import Union
from prettytable import PrettyTable

from ase import Atoms

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from autocat.learning.featurizers import Featurizer


class PredictorError(Exception):
    pass


class Predictor:
    def __init__(
        self, regressor=None, featurizer: Featurizer = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        regressor:
            Regressor object that can be used to make predictions
            (e.g. from scikit-learn) with `fit` and `predict` methods.
            **N.B**: If you want to make any changes to the parameters
            of this object after instantiation, please do so as follows:
            `predictor.regressor = updated_regressor`

        featurizer:
            `Featurizer` to be used for featurizing the structures
            when training and predicting.
            **N.B**: If you want to make any changes to the parameters
            of this object after instantiation, please do so as follows:
            `predictor.featurizer = updated_featurizer`

        """
        self.is_fit = False

        self._regressor = RandomForestRegressor()
        self.regressor = regressor

        self._featurizer = Featurizer()
        self.featurizer = featurizer

    def __repr__(self) -> str:
        pt = PrettyTable()
        pt.field_names = ["", "Predictor"]
        regressor_name = type(self.regressor)
        pt.add_row(["regressor", regressor_name])
        pt.add_row(["is fit?", self.is_fit])
        feat_str = str(self.featurizer)
        return str(pt) + "\n" + feat_str

    @property
    def regressor(self):
        return self._regressor

    @regressor.setter
    def regressor(self, regressor):
        if regressor is not None:
            self._regressor = copy.deepcopy(regressor)
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

    @property
    def featurizer(self):
        return self._featurizer

    @featurizer.setter
    def featurizer(self, featurizer):
        if featurizer is not None and isinstance(featurizer, Featurizer):
            self._featurizer = copy.deepcopy(featurizer)
            if self.is_fit:
                self.is_fit = False
                self.X_ = None
                self.y_ = None

    def copy(self):
        """
        Returns a copy
        """
        acp = self.__class__(regressor=self.regressor, featurizer=self.featurizer,)
        acp.is_fit = self.is_fit

        return acp

    def to_jsonified_dict(self) -> Dict:
        featurizer_dict = self.featurizer.to_jsonified_dict()
        regressor = self.regressor
        name_string = regressor.__class__.__name__
        module_string = regressor.__module__
        try:
            kwargs = regressor.get_params()
            _ = json.dumps(kwargs)
        except TypeError:
            print("Warning: kwargs not saved")
            kwargs = None
        return {
            "featurizer": featurizer_dict,
            "regressor": {
                "name_string": name_string,
                "module_string": module_string,
                "kwargs": kwargs,
            },
        }

    def write_json_to_disk(self, write_location: str = ".", json_name: str = None):
        """
        Writes `Predictor` to disk as a json
        """
        jsonified_list = self.to_jsonified_dict()

        if json_name is None:
            json_name = "predictor.json"

        json_path = os.path.join(write_location, json_name)

        with open(json_path, "w") as f:
            json.dump(jsonified_list, f)

    @staticmethod
    def from_jsonified_dict(all_data: Dict):
        # get regressor
        if all_data.get("regressor") is None:
            # allow not providing regressor (will use default)
            regressor = None
        elif not (
            isinstance(all_data.get("regressor"), dict)
            and all_data["regressor"].get("module_string") is not None
            and all_data["regressor"].get("name_string") is not None
        ):
            # check regressor is provided in the correct form
            msg = f"regressor must be provided\
                 in the form {{'module_string': module name, 'name_string': class name}},\
                 got {all_data.get('featurizer_class')}"
            raise PredictorError(msg)
        else:
            name_string = all_data["regressor"].get("name_string")
            module_string = all_data["regressor"].get("module_string")
            kwargs = all_data["regressor"].get("kwargs")
            mod = importlib.import_module(module_string)
            regressor_class = getattr(mod, name_string)
            if kwargs is not None:
                regressor = regressor_class(**kwargs)
            else:
                regressor = regressor_class()

        # get featurizer
        featurizer = Featurizer.from_jsonified_dict(all_data.get("featurizer", {}))
        return Predictor(regressor=regressor, featurizer=featurizer)

    @staticmethod
    def from_json(json_name: str):
        with open(json_name, "r") as f:
            all_data = json.load(f)
        return Predictor.from_jsonified_dict(all_data=all_data)

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
