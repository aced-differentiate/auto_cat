import numpy as np
import os
import json
from joblib import Parallel, delayed

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms
from ase.io import write as ase_write
from scipy import stats

from autocat.perturbations import generate_perturbed_dataset
from autocat.learning.predictors import AutoCatPredictor

Array = List[float]


class AutoCatSequentialLearningError(Exception):
    pass


def multiple_sequential_learning_runs(
    predictor: AutoCatPredictor,
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

    predictor:
        AutoCatPredictor object to be used for fitting and prediction

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
                predictor=predictor,
                training_base_structures=training_base_structures,
                **sl_kwargs,
            )
            for i in range(number_of_runs)
        )

    else:
        runs_history = [
            simulated_sequential_learning(
                predictor=predictor,
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
    predictor: AutoCatPredictor,
    all_training_structures: List[Atoms],
    all_training_y: np.ndarray,
    init_training_size: int = 10,
    testing_structures: List[Atoms] = None,
    testing_y: np.ndarray = None,
    acquisition_function: str = "MLI",
    batch_size_to_add: int = 1,
    number_of_sl_loops: int = 100,
    target_min: float = None,
    target_max: float = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok_structures: bool = False,
):
    """
    Conducts a simulated sequential learning loop given
    a set of base training structures.

    Parameters
    ----------

    predictor:
        AutoCatPredictor object to be used for fitting and prediction

    training_base_structures:
        List of Atoms objects for all base structures to be perturbed for training
        and candidate selection upon each loop

    testing_base_structures:
        List of Atoms objects for base structures that are

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

    if init_training_size > len(all_training_structures):
        msg = f"Initial training size ({init_training_size}) larger than design space ({len(all_training_structures)})"
        raise AutoCatSequentialLearningError(msg)

    # generate initial training set
    train_idx = np.zeros(len(all_training_structures), dtype=bool)
    train_idx[
        np.random.choice(
            len(all_training_structures), init_training_size, replace=False
        )
    ] = 1
    train_history = [train_idx]

    # fit on initial training set
    predictor.fit(all_training_structures[train_idx], all_training_y[train_idx])

    pred_history = []
    unc_history = []
    mae_train_history = []
    rmse_train_history = []
    mae_test_history = []
    rmse_test_history = []
    test_pred_history = []
    test_unc_history = []
    ctr = 0
    while ctr < number_of_sl_loops:
        ctr += 1
        print(f"Sequential Learning Iteration #{ctr}")
        # select new candidates to predict on

        # make predictions
        _preds, _uncs = predictor.predict(all_training_structures)

        # get scores on full training set
        mae_train_history.append(
            predictor.score(all_training_structures, all_training_y)
        )
        rmse_train_history.append(
            predictor.score(all_training_structures, all_training_y, metric="rmse")
        )

        # get scores on test perturbations
        if testing_structures is not None:
            mae_test_score, test_preds, test_unc = predictor.score(
                testing_structures, testing_y, return_predictions=True
            )
            mae_test_history.append(mae_test_score)
            test_pred_history.append([p.tolist() for p in test_preds])
            test_unc_history.append([u.tolist() for u in test_unc])
            rmse_test_history.append(
                predictor.score(testing_structures, testing_y, metric="rmse")
            )

        unc_history.append(_uncs)
        pred_history.append(_preds)

        # select next candidate(s)
        next_candidate_idx, max_scores, aq_scores = choose_next_candidate(
            all_training_y,
            train_idx,
            _preds,
            _uncs,
            acquisition_function,
            batch_size_to_add,
            target_min,
            target_max,
        )

        # add new perturbed struct to training set
        train_pert_structures.extend(next_candidate_struct)
        train_pert_corr_list = np.concatenate(
            (
                train_pert_corr_list,
                np.array([new_pert_corr_list[idx] for idx in high_unc_idx]),
            )
        )

        # keeps all candidate structures added for each loop
        selected_candidate_history.append(next_candidate_struct)

        # keeps as lists to make writing to disk easier
        candidate_pred_corrs_history.append(
            [p.tolist() for p in [_pred_corrs[idx] for idx in high_unc_idx]]
        )
        candidate_real_corrs_history.append(
            [new_pert_corr_list[idx].tolist() for idx in high_unc_idx]
        )
        predictor.fit(train_pert_structures, train_pert_corr_list)
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
        sl_dict["test_real_corrections"] = [t.tolist() for t in test_pert_corr_list]
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


def choose_next_candidate(
    labels: Array = None,
    train_idx: Array = None,
    pred: Array = None,
    unc: Array = None,
    aq: str = "MLI",
    num_candidates_to_pick: int = None,
    target_min: float = None,
    target_max: float = None,
):
    """
    Chooses the next candidate(s) from a given acquisition function

    Parameters
    ----------

    labels:
        Array of the labels for the data

    train_idx:
        Indices of all data entries already in the training set
        Default: consider entire training set

    pred:
        Predictions for all structures in the dataset

    unc:
        Uncertainties for all structures in the dataset

    aq:
        Acquisition function to be used to select the next candidates
        Options
        - MLI: maximum likelihood of improvement (default)
        - Random
        - MU: maximum uncertainty

    num_candidates_to_pick:
        Number of candidates to choose from the dataset

    target_min:
        Minimum target value to optimize for

    target_max:
        Maximum target value to optimize for

    Returns
    -------

    parent_idx:
        Index/indices of the selected candidates

    max_scores:
        Maximum scores (corresponding to the selected candidates for given `aq`)

    aq_scores:
        Calculated scores based on the selected `aq` for the entire training set
    """
    if aq == "Random":
        if labels is None:
            msg = "For aq = 'Random', the labels must be supplied"
            raise AutoCatSequentialLearningError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(labels), dtype=bool)

        next_idx = np.random.choice(
            len(labels[~train_idx]), size=num_candidates_to_pick, replace=False
        )
        if num_candidates_to_pick is None:
            next_idx = np.array([next_idx])
        parent_idx = np.arange(labels.shape[0])[~train_idx][next_idx]

        max_scores = []
        aq_scores = []

    elif aq == "MU":
        if unc is None:
            msg = "For aq = 'MU', the uncertainties must be supplied"
            raise AutoCatSequentialLearningError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(unc), dtype=bool)

        if num_candidates_to_pick is None:
            next_idx = np.array([np.argmax(unc[~train_idx])])
            max_scores = [np.max(unc[~train_idx])]

        else:
            next_idx = np.argsort(unc[~train_idx])[-num_candidates_to_pick:]
            sorted_array = unc[~train_idx][next_idx]
            max_scores = list(sorted_array[-num_candidates_to_pick:])
        parent_idx = np.arange(unc.shape[0])[~train_idx][next_idx]
        aq_scores = unc

    elif aq == "MLI":
        if unc is None or pred is None:
            msg = "For aq = 'MLI', both uncertainties and predictions must be supplied"
            raise AutoCatSequentialLearningError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(unc), dtype=bool)

        aq_scores = np.array(
            [
                get_overlap_score(mean, std, x2=target_max, x1=target_min)
                for mean, std in zip(pred, unc)
            ]
        )
        if num_candidates_to_pick is None:
            next_idx = np.array([np.argmax(aq_scores[~train_idx])])
            max_scores = [np.max(aq_scores[~train_idx])]

        else:
            next_idx = np.argsort(aq_scores[~train_idx])[-num_candidates_to_pick:]
            sorted_array = aq_scores[~train_idx][next_idx]
            max_scores = list(sorted_array[-num_candidates_to_pick:])
        parent_idx = np.arange(aq_scores.shape[0])[~train_idx][next_idx]

    else:
        msg = f"Acquisition function {aq} is not supported"
        raise NotImplementedError(msg)

    return parent_idx, max_scores, aq_scores


def get_overlap_score(mean: float, std: float, x2: float = None, x1: float = None):
    """Calculate overlap score given targets x2 (max) and x1 (min)"""
    if x1 is None and x2 is None:
        msg = "Please specify at least either a minimum or maximum target for MLI"
        raise AutoCatSequentialLearningError(msg)

    if x1 is None:
        x1 = -np.inf

    if x2 is None:
        x2 = np.inf

    norm_dist = stats.norm(loc=mean, scale=std)
    return norm_dist.cdf(x2) - norm_dist.cdf(x1)
