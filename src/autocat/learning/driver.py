from typing import List
from typing import Dict
from typing import Union

from pypif.pif import System
from citrination_client import CitrinationClient, data
from citrination_client.search import Filter
from citrination_client.search import FieldQuery
from citrination_client.search import DatasetQuery
from citrination_client.search import DataQuery
from citrination_client.search import PifSystemQuery
from citrination_client.search import PifSearchHit
from citrination_client.search import PifSystemReturningQuery

from autocat.learning.predictors import AutoCatPredictor
from autocat.learning.sequential import AutoCatDesignSpace
from autocat.learning.sequential import AutoCatSequentialLearner
from autocat.learning.sequential import AutoCatSequentialLearningError


def AutoCatSequentialDriverError(Exception):
    pass


class AutoCatSequentialDriver(object):
    """Driver for sequential learning."""

    def __init__(
        self, dataset_id: int = None, current_learner: AutoCatSequentialLearner = None,
    ):
        """Constructor docstring goes here."""
        self._dataset_id = None
        self.dataset_id = dataset_id

        self._current_learner = None
        self.current_learner = current_learner

    @property
    def dataset_id(self):
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        if dataset_id is None:
            msg = "Citrination dataset ID is not valid (None)"
            raise AutoCatSequentialDriverError(msg)
        self._dataset_id = dataset_id

    @property
    def current_learner(self):
        return self._current_learner

    @current_learner.setter
    def current_learner(self, current_learner):
        if isinstance(current_learner, AutoCatSequentialLearner):
            self._current_learner = current_learner
        elif isinstance(current_learner, str):
            self._current_learner = AutoCatSequentialLearner.from_json(current_learner)
        else:
            msg = "Input AutoCat sequential learner is not valid"
            raise AutoCatSequentialDriverError(msg)

    def drive(self, new_design_space: AutoCatDesignSpace = None):
        self.current_learner.iterate(new_design_space)
        # self.current_learner.candidate_structures
        # write new input files
