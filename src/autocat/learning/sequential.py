import numpy as np
import os
import json
from joblib import Parallel, delayed

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms
from ase.io import write as ase_write

from autocat.perturbations import generate_perturbed_dataset
from autocat.learning.predictors import AutoCatStructureCorrector

Array = List[float]


class AutoCatSequentialLearningError(Exception):
    pass


def multiple_sequential_learning_runs(
    structure_corrector: AutoCatStructureCorrector,
    training_base_structures: List[Atoms],
    number_of_runs: int = 5,
    number_parallel_jobs: int = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok_structures: bool = False,
    **sl_kwargs,
):
    """
    Conducts multiple sequential learning runs

    Parameters
    ----------

    structure_corrector:
        AutoCatStructureCorrector object to be used for fitting and prediction

    training_base_structures:
        List of Atoms objects for all base structures to be perturbed for training
        and candidate selection upon each loop

    number_of_runs:
        Integer of number of runs to be done

    number_parallel_jobs:
        Integer giving the number of cores to be paralellized across
        using `joblib`

    write_to_disk:
        Boolean specifying whether runs history should be written to disk as a json.
        Defaults to False.

    write_location:
        String with the location where runs history should be written to disk.

    dirs_exist_ok_structures:
        Boolean indicating if existing candidate structure files can be overwritten

    Returns
    -------

    run_history:
        List of dictionaries generated for each run containing info
        about that run such as mae history, rmse history, etc..
    """
    if number_parallel_jobs is not None:
        runs_history = Parallel(n_jobs=number_parallel_jobs)(
            delayed(simulated_sequential_learning)(
                structure_corrector=structure_corrector,
                training_base_structures=training_base_structures,
                **sl_kwargs,
            )
            for i in range(number_of_runs)
        )

    else:
        runs_history = [
            simulated_sequential_learning(
                structure_corrector=structure_corrector,
                training_base_structures=training_base_structures,
                **sl_kwargs,
            )
            for i in range(number_of_runs)
        ]

    if write_to_disk:
        if not os.path.isdir(write_location):
            os.makedirs(write_location)
        json_write_path = os.path.join(write_location, "sl_runs_history.json")
        data_runs_history = []
        for r_idx, run in enumerate(runs_history):
            # get dict excluding selected candidate structures
            j_dict = {
                key: run[key] for key in run if key != "selected_candidate_history"
            }
            data_runs_history.append(j_dict)
            # write out selected candidates for each iteration of each run
            c_hist = run["selected_candidate_history"]
            traj_file_path = os.path.join(
                write_location, f"candidate_structures/run{r_idx+1}",
            )
            os.makedirs(traj_file_path, exist_ok=dirs_exist_ok_structures)
            for i, c in enumerate(c_hist):
                traj_filename = os.path.join(traj_file_path, f"sl_iter{i+1}.traj")
                ase_write(traj_filename, c)
                print(
                    f"Selected SL Candidates for run {r_idx+1}, iteration {i+1} written to {traj_file_path}"
                )
        with open(json_write_path, "w") as f:
            json.dump(data_runs_history, f)
        print(f"SL histories written to {json_write_path}")

    return runs_history


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
    dirs_exist_ok_structures: bool = False,
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
        Boolean specifying whether the sl dictionary should be written to disk as a json.
        Defaults to False.

    write_location:
        String with the location where sl_dict should be written to disk.

    dirs_exist_ok_structures:
        Boolean indicating if existing candidate structure files can be overwritten

    Returns
    -------

    sl_dict:
        Dictionary containing histories of different quantities throughout
        the calculation:
        - candidate maximum uncertainty history
        - candidate full uncertainty history
        - predicted corrections training history
        - real corrections training history
        - mae training history
        - rmse training history
        - mae testing history (if test structures given)
        - rmse testing history (if test structures given)
        - testing predictions (if test structures given)
        - testing uncertainties (if test structures given)
        For the corrections histories, the dimensions are as follows:
        num of loops -> num of candidates added -> corrections applied
    """
    if (
        batch_num_of_perturbations_per_base_structure * len(training_base_structures)
        < batch_size_to_add
    ):
        msg = "Batch size to add must be less than the number of candidates generated"
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

    candidate_full_unc_history = []
    candidate_max_unc_history = []
    candidate_pred_corrs_history = []
    candidate_real_corrs_history = []
    mae_train_history = []
    rmse_train_history = []
    mae_test_history = []
    rmse_test_history = []
    selected_candidate_history = []
    test_preds_history = []
    test_unc_history = []
    ctr = 0
    while len(candidate_max_unc_history) < number_of_sl_loops:
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
            mae_test_score, test_preds, test_unc = structure_corrector.score(
                test_pert_structures, test_pert_corr_list, return_predictions=True
            )
            mae_test_history.append(mae_test_score)
            test_preds_history.append([p.tolist() for p in test_preds])
            test_unc_history.append([u.tolist() for u in test_unc])
            rmse_test_history.append(
                structure_corrector.score(
                    test_pert_structures, test_pert_corr_list, metric="rmse"
                )
            )

        candidate_full_unc_history.append(_uncs)

        # find candidate with highest uncertainty
        high_unc_idx = np.argsort(_uncs)[-batch_size_to_add:]
        candidate_max_unc_history.append(_uncs[high_unc_idx])

        next_candidate_struct = [new_pert_structs[idx] for idx in high_unc_idx]
        # add new perturbed struct to training set
        train_pert_structures.extend(next_candidate_struct)
        train_pert_corr_list.extend([new_pert_corr_list[idx] for idx in high_unc_idx],)

        # keeps all candidate structures added for each loop
        selected_candidate_history.append(next_candidate_struct)

        # keeps as lists to make writing to disk easier
        candidate_pred_corrs_history.append(
            [p.tolist() for p in [_pred_corrs[idx] for idx in high_unc_idx]]
        )
        candidate_real_corrs_history.append(
            [new_pert_corr_list[idx].tolist() for idx in high_unc_idx]
        )
        structure_corrector.fit(
            train_pert_structures, corrections_list=train_pert_corr_list
        )
    sl_dict = {
        "candidate_max_unc_history": [mu.tolist() for mu in candidate_max_unc_history],
        "candidate_full_unc_history": [
            unc.tolist() for unc in candidate_full_unc_history
        ],
        "candidate_pred_corrs_history": candidate_pred_corrs_history,
        "candidate_real_corrs_history": [r for r in candidate_real_corrs_history],
        "selected_candidate_history": selected_candidate_history,
        "mae_train_history": mae_train_history,
        "rmse_train_history": rmse_train_history,
    }

    if testing_base_structures is not None:
        sl_dict["mae_test_history"] = mae_test_history
        sl_dict["rmse_test_history"] = rmse_test_history
        sl_dict["test_preds_history"] = test_preds_history
        sl_dict["test_unc_history"] = test_unc_history

    if write_to_disk:
        if not os.path.isdir(write_location):
            os.makedirs(write_location)
        json_write_path = os.path.join(write_location, "sl_dict.json")
        data_sl_dict = {
            key: sl_dict[key] for key in sl_dict if key != "selected_candidate_history"
        }
        with open(json_write_path, "w") as f:
            json.dump(data_sl_dict, f)
        print(f"SL dictionary written to {json_write_path}")
        candidate_struct_hist = sl_dict["selected_candidate_history"]
        traj_file_path = os.path.join(write_location, f"candidate_structures")
        os.makedirs(traj_file_path, exist_ok=dirs_exist_ok_structures)
        for i, c in enumerate(candidate_struct_hist):
            traj_filename = os.path.join(traj_file_path, f"sl_iter{i+1}.traj")
            ase_write(traj_filename, c)
            print(
                f"Selected SL Candidates for iteration {i+1} written to {traj_file_path}"
            )

    return sl_dict
