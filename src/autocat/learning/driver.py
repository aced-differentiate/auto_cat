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
from autocat.learning.sequential import (
    AutoCatDesignSpace,
    AutoCatSequentialLearner,
    AutoCatSequentialLearningError,
)


def AutoCatSequentialDriverError(Exception):
    pass


def get_pif_query(
    dataset_id: int = None, tags: List[str] = None
) -> PifSystemReturningQuery:
    """
    Creates a Citrination query object that can be used to search for PIFs
    with the specified tags in the specified dataset.
    """
    dataset_q = DatasetQuery(id=Filter(equal=dataset_id))
    tags_q = [FieldQuery(filter=Filter(equal=tag), extract_all=True) for tag in tags]
    system_q = PifSystemQuery(tags=tags_q)
    data_q = DataQuery(dataset=dataset_q, system=system_q)
    return PifSystemReturningQuery(query=data_q)


def query_pifs_from_citrination(
    client: CitrinationClient = None,
    dataset_id: int = None,
    host: str = None,
    dopant: str = None,
    termination: str = None,
) -> List[PifSearchHit]:
    """
    Queries the specified Citrination dataset for a specified SAA catalyst
    system (defined by the host lattice, single atom dopant, and surface
    termination).
    """
    print(f"Querying dataset (Citrination ID {dataset_id}) for:")
    print(f"  Host = {host}")
    print(f"  SA Dopant = {dopant}")
    print(f"  Termination = ({termination})")
    tags = [
        f"host:{host}",
        f"sa-dopant:{dopant}",
        f"termination:{termination}",
    ]
    query = get_pif_query(dataset_id=dataset_id, tags=tags)
    pif_search_result = client.search.pif_search(query)
    print(f"Number of search hits: {pif_search_result.total_num_hits}")
    print("")
    return pif_search_result.hits


def get_system_total_energy(system: System) -> float:
    """Returns the total energy value in the input PIF object."""
    total_energy_filter = filter(lambda x: "Total Energy" in x.name, system.properties)
    return list(total_energy_filter)[0].scalars[0].value


def get_reference_energies(
    pif_search_hits: List[PifSearchHit], references: List[str] = None
) -> Dict[str, List[Dict[str, float]]]:
    """
    For each specified reference, filters the total energy from the input
    list of PIFs queried from Citrination, and returns them as lists in a
    dictionary.
    """
    reference_energies = {}
    for reference in references:
        f_search_hits = filter(
            lambda x: f"reference:{reference}" in x.system.tags, pif_search_hits
        )
        reference_energies[reference] = [
            {"total_energy": get_system_total_energy(f_search_hit.system)}
            for f_search_hit in f_search_hits
        ]
    return reference_energies


def get_configuration_tags(system: System) -> List[str]:
    """Returns site configuration tags in the input PIF object as a list."""
    configuration_tags = filter(
        lambda x: "site-type" in x or "coordinates" in x, system.tags
    )
    return list(configuration_tags)


def get_intermediate_energies(
    pif_search_hits: List[PifSearchHit], intermediates: List[str] = None
) -> Dict[str, List[Dict[str, Union[List[str], float]]]]:
    """
    For each specified intermediate species, filters the total energy and
    configuration tags from the input list of PIFs queried from Citrination,
    and returns them as lists of dictionaries.
    """
    intermediate_energies = {}
    for intermediate in intermediates:
        f_search_hits = filter(
            lambda x: f"intermediate:{intermediate}" in x.system.tags, pif_search_hits
        )
        intermediate_energies[intermediate] = [
            {
                "configuration_tags": get_configuration_tags(f_search_hit.system),
                "total_energy": get_system_total_energy(f_search_hit.system),
            }
            for f_search_hit in f_search_hits
        ]
    return intermediate_energies


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
