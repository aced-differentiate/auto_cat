import numpy as np
import os
import json

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms

from autocat.perturbations import generate_perturbed_dataset
from autocat.learning.predictors import AutoCatStructureCorrector

Array = List[float]


class AutoCatSequentialLearningError(Exception):
    pass


def simulated_sequential_learning(
    structure_corrector: AutoCatStructureCorrector,
    training_base_structures: List[Atoms],
    testing_base_structures: List[Atoms] = None,
    minimum_perturbation_distance: float = 0.01,
    maximum_perturbation_distance: float = 0.75,
    initial_num_of_perturbations_per_base_structure: int = None,
    batch_num_of_perturbations_per_base_structure: int = 2,
    batch_size_to_add: int = 1,
    number_of_sl_loops: int = 100,
    write_to_disk: bool = False,
    write_location: str = ".",
):
    """
    Conducts a simulated sequential learning loop given
    a set of base structures. For each loop, new perturbations
    are generated. Maximum Uncertainty is used as the acquisition function

    Parameters
    ----------

    structure_corrector:
        AutoCatStructureCorrector object to be used for fitting and prediction

    training_base_structures:
        List of Atoms objects for all base structures to be perturbed for training
        and candidate selection upon each loop

    testing_base_structures:
        List of Atoms objects for base structures that are

    minimum_perturbation_distance:
        Float of minimum acceptable perturbation distance
        Default: 0.01 Angstrom

    maximum_perturbation_distance:
        Float of maximum acceptable perturbation distance
        Default: 0.75 Angstrom

    initial_num_of_perturbations_per_base_structure:
        Integer giving the number of perturbations to generate initially
        for each of the given base structures for initial training purposes.
        Default: uses same value as `batch_num_of_perturbations_per_base_structure`

    batch_num_of_perturbations_per_base_structure:
        Integer giving the number of perturbations to generate
        on each loop when finding the next candidate

    batch_size_to_add:
        Integer giving the number of candidates to be added
        to the training set on each loop. (ie. N candidates with
        the highest uncertainties added for each iteration)
        Default: 1 (ie. adds candidate with max unc on each loop)

    number_of_sl_loops:
        Integer specifying the number of sequential learning loops to be conducted

    write_to_disk:
        Boolean specifying whether X should be written to disk as a json.
        Defaults to False.

    write_location:
        String with the location where X should be written to disk.

    Returns
    -------

    sl_dict:
        Dictionary containing histories of different quantities throughout
        the calculation:
        - maximum uncertainty history
        - full uncertainty history
        - predicted corrections training history
        - real corrections training history
        - mae training history
        - rmse training history
        - mae testing history (if test structures given)
        - rmse testing history (if test structures given)
        For the corrections histories, the dimensions are as follows:
        num of loops -> num of candidates added -> corrections applied
    """
    if (
        batch_num_of_perturbations_per_base_structure * len(training_base_structures)
        < batch_size_to_add
    ):
        msg = (
            "Batch size to add must be greater than the number of candidates generated"
        )
        raise AutoCatSequentialLearningError(msg)

    if initial_num_of_perturbations_per_base_structure is None:
        initial_num_of_perturbations_per_base_structure = (
            batch_num_of_perturbations_per_base_structure
        )

    initial_pert_dataset = generate_perturbed_dataset(
        training_base_structures,
        num_of_perturbations=initial_num_of_perturbations_per_base_structure,
        maximum_perturbation_distance=maximum_perturbation_distance,
        minimum_perturbation_distance=minimum_perturbation_distance,
    )

    train_pert_structures = initial_pert_dataset["collected_structures"]
    train_pert_corr_list = initial_pert_dataset["corrections_list"]

    structure_corrector.fit(
        train_pert_structures, corrections_list=train_pert_corr_list
    )

    if testing_base_structures is not None:
        test_pert_dataset = generate_perturbed_dataset(
            testing_base_structures,
            num_of_perturbations=initial_num_of_perturbations_per_base_structure,
            maximum_perturbation_distance=maximum_perturbation_distance,
            minimum_perturbation_distance=minimum_perturbation_distance,
        )
        test_pert_structures = test_pert_dataset["collected_structures"]
        test_pert_corr_list = test_pert_dataset["corrections_list"]

    validation_pert_dataset = generate_perturbed_dataset(
        training_base_structures,
        num_of_perturbations=initial_num_of_perturbations_per_base_structure,
        maximum_perturbation_distance=maximum_perturbation_distance,
        minimum_perturbation_distance=minimum_perturbation_distance,
    )
    validation_pert_structures = validation_pert_dataset["collected_structures"]
    validation_pert_corr_list = validation_pert_dataset["corrections_list"]

    full_unc_history = []
    max_unc_history = []
    pred_corrs_history = []
    real_corrs_history = []
    mae_train_history = []
    rmse_train_history = []
    mae_test_history = []
    rmse_test_history = []
    ctr = 0
    while len(max_unc_history) < number_of_sl_loops:
        ctr += 1
        print(f"Sequential Learning Iteration #{ctr}")
        # generate new perturbations to predict on
        new_perturbations = generate_perturbed_dataset(
            training_base_structures,
            num_of_perturbations=batch_num_of_perturbations_per_base_structure,
            maximum_perturbation_distance=maximum_perturbation_distance,
            minimum_perturbation_distance=minimum_perturbation_distance,
        )
        new_pert_structs = new_perturbations["collected_structures"]
        new_pert_corr_list = new_perturbations["corrections_list"]

        # make predictions
        _pred_corrs, _pred_corr_structs, _uncs = structure_corrector.predict(
            new_pert_structs
        )

        # get scores on new perturbations (training)
        mae_train_history.append(
            structure_corrector.score(
                validation_pert_structures, validation_pert_corr_list
            )
        )
        rmse_train_history.append(
            structure_corrector.score(
                validation_pert_structures, validation_pert_corr_list, metric="rmse"
            )
        )

        # get scores on test perturbations
        if testing_base_structures is not None:
            mae_test_history.append(
                structure_corrector.score(test_pert_structures, test_pert_corr_list)
            )
            rmse_test_history.append(
                structure_corrector.score(
                    test_pert_structures, test_pert_corr_list, metric="rmse"
                )
            )

        full_unc_history.append(_uncs)

        # find candidate with highest uncertainty
        high_unc_idx = np.argsort(_uncs)[-batch_size_to_add:]
        max_unc_history.append(_uncs[high_unc_idx])

        next_candidate_struct = [new_pert_structs[idx] for idx in high_unc_idx]
        # add new perturbed struct to training set
        train_pert_structures.extend(next_candidate_struct)
        train_pert_corr_list.extend([new_pert_corr_list[idx] for idx in high_unc_idx],)
        # keeps as lists to make writing to disk easier
        pred_corrs_history.append(
            [p.tolist() for p in [_pred_corrs[idx] for idx in high_unc_idx]]
        )
        real_corrs_history.append(
            [new_pert_corr_list[idx].tolist() for idx in high_unc_idx]
        )
        structure_corrector.fit(
            train_pert_structures, corrections_list=train_pert_corr_list
        )
    sl_dict = {
        "max_unc_history": [mu.tolist() for mu in max_unc_history],
        "full_unc_history": [unc.tolist() for unc in full_unc_history],
        "pred_corrs_history": pred_corrs_history,
        "real_corrs_history": [r for r in real_corrs_history],
        "mae_train_history": mae_train_history,
        "rmse_train_history": rmse_train_history,
    }

    if testing_base_structures is not None:
        sl_dict["mae_test_history"] = mae_test_history
        sl_dict["rmse_test_history"] = rmse_test_history

    if write_to_disk:
        if not os.path.isdir(write_location):
            os.makedirs(write_location)
        write_path = os.path.join(write_location, "sl_dict.json")
        with open(write_path, "w") as f:
            json.dump(sl_dict, f)
        print(f"SL dictionary written to {write_path}")

    return sl_dict
