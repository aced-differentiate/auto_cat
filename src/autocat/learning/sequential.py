import numpy as np

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms

from autocat.perturbations import generate_perturbed_dataset
from autocat.learning.predictors import AutoCatStructureCorrector

Array = List[float]


def simulated_sequential_learning(
    structure_corrector: AutoCatStructureCorrector,
    base_structures: List[Atoms],
    minimum_perturbation_distance: float = 0.1,
    maximum_perturbation_distance: float = 1.0,
    initial_num_of_perturbations_per_base_structure: int = None,
    batch_num_of_perturbations_per_base_structure: int = 2,
    number_of_sl_loops: int = 100,
):
    """
    Conducts a simulated sequential learning loop given
    a set of base structures. For each loop, new perturbations
    are generated. Maximum Uncertainty is used as the acquisition function

    Parameters
    ----------

    structure_corrector:
        AutoCatStructureCorrector object to be used for fitting and prediction

    base_structures:
        List of Atoms objects for all base structures to be perturbed for training
        and testing

    minimum_perturbation_distance:
        Float of minimum acceptable perturbation distance
        Default: 0.1 Angstrom

    maximum_perturbation_distance:
        Float of maximum acceptable perturbation distance
        Default: 1.0 Angstrom

    initial_num_of_perturbations_per_base_structure:
        Integer giving the number of perturbations to generate initially
        for each of the given base structures for initial training purposes.
        Default: uses same value as `batch_num_of_perturbations_per_base_structure`

    batch_num_of_perturbations_per_base_structure:
        Integer giving the number of perturbations to generate
        on each loop when finding the next candidate

    number_of_sl_loops:
        Integer specifying the number of sequential learning loops to be conducted

    Returns
    -------
    """
    if initial_num_of_perturbations_per_base_structure is None:
        initial_num_of_perturbations_per_base_structure = (
            batch_num_of_perturbations_per_base_structure
        )

    initial_pert_dataset = generate_perturbed_dataset(
        base_structures,
        num_of_perturbations=initial_num_of_perturbations_per_base_structure,
        maximum_perturbation_distance=maximum_perturbation_distance,
        minimum_perturbation_distance=minimum_perturbation_distance,
    )

    train_pert_structures = initial_pert_dataset["collected_structures"]
    train_pert_coll_matr = initial_pert_dataset["collected_matrices"]

    structure_corrector.fit(train_pert_structures, train_pert_coll_matr)

    full_unc_history = []
    max_unc_history = []
    pred_corr_matr_history = []
    while len(max_unc_history) < number_of_sl_loops:
        # generate new perturbations to predict on
        new_perturbations = generate_perturbed_dataset(
            base_structures,
            num_of_perturbations=batch_num_of_perturbations_per_base_structure,
            maximum_perturbation_distance=maximum_perturbation_distance,
            minimum_perturbation_distance=minimum_perturbation_distance,
        )
        new_pert_structs = new_perturbations["collected_structures"]
        new_pert_coll_matr = new_perturbations["collected_matrices"]

        # make predictions
        _pred_corr_matr, _pred_corr_structs, _uncs = structure_corrector.predict(
            new_pert_structs
        )
        full_unc_history.append(_uncs)
        pred_corr_matr_history.append(_pred_corr_matr)

        # find candidate with highest uncertainty
        high_unc_idx = np.argmax(_uncs)
        max_unc_history.append(_uncs[high_unc_idx])

        next_candidate_struct = new_pert_structs[high_unc_idx]
        # add new perturbed struct to training set
        train_pert_structures.append(next_candidate_struct)
        train_pert_coll_matr = np.concatenate(
            (train_pert_coll_matr, new_pert_coll_matr[high_unc_idx].reshape(1, -1)),
            axis=0,
        )
        structure_corrector.fit(train_pert_structures, train_pert_coll_matr)
    return {
        "max_unc_history": max_unc_history,
        "full_unc_history": full_unc_history,
        "pred_corr_matr_history": pred_corr_matr_history,
    }
