import numpy as np
import os
import json
from joblib import Parallel, delayed

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms
from scipy import stats

from autocat.learning.predictors import AutoCatPredictor
from autocat.data.hhi import HHI_PRODUCTION
from autocat.data.hhi import HHI_RESERVES

Array = List[float]


class AutoCatSequentialLearningError(Exception):
    pass


def multiple_simulated_sequential_learning_runs(
    number_of_runs: int = 5,
    number_parallel_jobs: int = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    sl_kwargs=None,
):
    """
    Conducts multiple simulated sequential learning runs

    Parameters
    ----------

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

    sl_kwargs:
        Mapping of keywords for `simulated_sequential_learning`.
        Note: Do not use the `write_to_disk` keyword here

    Returns
    -------

    run_history:
        List of dictionaries generated for each run containing info
        about that run such as mae history, rmse history, etc..
    """
    if sl_kwargs is None:
        sl_kwargs = {}

    if number_parallel_jobs is not None:
        runs_history = Parallel(n_jobs=number_parallel_jobs)(
            delayed(simulated_sequential_learning)(**sl_kwargs,)
            for i in range(number_of_runs)
        )

    else:
        runs_history = [
            simulated_sequential_learning(**sl_kwargs,) for i in range(number_of_runs)
        ]

    if write_to_disk:
        if not os.path.isdir(write_location):
            os.makedirs(write_location)
        json_write_path = os.path.join(write_location, "sl_runs_history.json")
        with open(json_write_path, "w") as f:
            json.dump(runs_history, f)
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
    number_of_sl_loops: int = None,
    target_min: float = None,
    target_max: float = None,
    write_to_disk: bool = False,
    write_location: str = ".",
):
    """
    Conducts a simulated sequential learning loop given
    a data set to explore. Can optionally provide a holdout
    test set that is never added to the training set
    to measure transferability at each iteration

    Parameters
    ----------

    predictor:
        AutoCatPredictor object to be used for fitting and prediction

    all_training_structures:
        List of all Atoms objects that make up the design space
        to be considered

    all_training_y:
        Labels corresponding to each structure in the design
        space to be explored (ie. correspond to
        `all_training_structures`)

    init_training_size:
        Size of the initial training set to be selected from
        the full space.
        Default: 10

    testing_structures:
        List of all Atoms objects to be excluded from the
        SL search (ie. structures to never be added to the set).
        Used for testing transferability of the model at each iteration

    testing_y:
        Labels for corresponding to `testing_structures`

    acquisition_function:
        Acquisition function to be used to determine candidates to be
        added to the training set at each iteration of the sl loop

    batch_size_to_add:
        Number of candidates to be added to the training set on each loop.
        (ie. N candidates with the highest uncertainties added
        for each iteration)
        Default: 1 (ie. adds candidate with max unc on each loop)

    number_of_sl_loops:
        Integer specifying the number of sequential learning loops to be conducted.
        This value cannot be greater than
        `(len(all_training_structures) - init_training_size)/batch_size_to_add`
        Default: maximum number of sl loops calculated above

    target_min:
        Label value that ideal candidates should be greater than
        Default: -inf

    target_max:
        Label value that ideal candidates should be less than
        Default: +inf

    write_to_disk:
        Boolean specifying whether the sl dictionary should be written to disk as a json.
        Defaults to False.

    write_location:
        String with the location where sl_dict should be written to disk.

    Returns
    -------

    sl_dict:
        Dictionary containing histories of different quantities throughout
        the calculation:
        - training_history: indices that are included in the training set
        - uncertainty_history: prediction uncertainties at each iteration
        - predicted_history: all predictions at each iteration
        - mae_train_history: mae scores on predicting training set
        - rmse_train_history: rmse scores on predicting training set
        - max_scores_history: max aq scores for candidate selection
        - aq_scores_history: all aq scores at each iteration
        If testing structures and labels given:
        - mae_test_history: mae scores on testing set
        - rmse_test_history: rmse scores on testing set
        - test_prediction_history: predictions on testing set
        - test_uncertainty_history: uncertainties on predicting on test set
    """

    if init_training_size > len(all_training_structures):
        msg = f"Initial training size ({init_training_size}) larger than design space ({len(all_training_structures)})"
        raise AutoCatSequentialLearningError(msg)

    max_num_sl_loops = int(
        np.ceil((len(all_training_structures) - init_training_size) / batch_size_to_add)
    )

    if number_of_sl_loops is None:
        number_of_sl_loops = max_num_sl_loops

    if number_of_sl_loops > max_num_sl_loops:
        msg = f"Number of SL loops ({number_of_sl_loops}) cannot be greater than ({max_num_sl_loops})"
        raise AutoCatSequentialLearningError(msg)

    # generate initial training set
    train_idx = np.zeros(len(all_training_structures), dtype=bool)
    train_idx[
        np.random.choice(
            len(all_training_structures), init_training_size, replace=False
        )
    ] = 1
    train_history = [train_idx.copy()]

    # fit on initial training set
    X = [s for s, i in zip(all_training_structures, train_idx) if i]
    y = all_training_y[train_idx]
    predictor.fit(X, y)

    pred_history = []
    unc_history = []
    max_scores_history = []
    aq_scores_history = []
    mae_train_history = []
    rmse_train_history = []
    mae_test_history = []
    rmse_test_history = []
    test_pred_history = []
    test_unc_history = []

    def _collect_pred_stats(
        all_training_structures, all_training_y, testing_structures, testing_y
    ):
        _preds, _uncs = predictor.predict(all_training_structures)

        # get scores on full training set
        mae_train_history.append(
            predictor.score(all_training_structures, all_training_y)
        )
        rmse_train_history.append(
            predictor.score(
                all_training_structures, all_training_y, metric="mse", squared=False
            )
        )

        # get scores on test perturbations
        if testing_structures is not None:
            if testing_y is None:
                msg = f"Labels for the test structures must be provided"
                raise AutoCatSequentialLearningError(msg)

            mae_test_score, test_preds, test_unc = predictor.score(
                testing_structures, testing_y, return_predictions=True
            )
            mae_test_history.append(mae_test_score)
            test_pred_history.append([p.tolist() for p in test_preds])
            test_unc_history.append([u.tolist() for u in test_unc])
            rmse_test_history.append(
                predictor.score(
                    testing_structures, testing_y, metric="mse", squared=False
                )
            )

        unc_history.append(_uncs)
        pred_history.append(_preds)
        return _preds, _uncs

    for i in range(number_of_sl_loops):
        print(f"Sequential Learning Iteration #{i+1}")
        # make predictions
        _preds, _uncs = _collect_pred_stats(
            all_training_structures, all_training_y, testing_structures, testing_y
        )

        # check that enough data pts left to adhere to batch_size_to_add
        # otherwise just add the leftover data
        if batch_size_to_add < len(all_training_y[~train_idx]):
            bsa = batch_size_to_add
        else:
            bsa = len(all_training_y[~train_idx])

        # select next candidate(s)
        next_candidate_idx, max_scores, aq_scores = choose_next_candidate(
            labels=all_training_y,
            train_idx=train_idx,
            pred=_preds,
            unc=_uncs,
            aq=acquisition_function,
            num_candidates_to_pick=bsa,
            target_min=target_min,
            target_max=target_max,
        )
        max_scores_history.append([int(i) for i in max_scores])
        aq_scores_history.append(aq_scores.tolist())

        # add next candidates to training set
        train_idx[next_candidate_idx] = True
        train_history.append(train_idx.copy())

        # retrain on training set with new additions
        X = [s for s, i in zip(all_training_structures, train_idx) if i]
        y = all_training_y[train_idx]
        predictor.fit(X, y)

    # make preds on final model
    _, _ = _collect_pred_stats(
        all_training_structures, all_training_y, testing_structures, testing_y
    )

    sl_dict = {
        "training_history": [th.tolist() for th in train_history],
        "uncertainty_history": [mu.tolist() for mu in unc_history],
        "prediction_history": [p.tolist() for p in pred_history],
        "mae_train_history": mae_train_history,
        "rmse_train_history": rmse_train_history,
        "max_scores_history": max_scores_history,
        "aq_scores_history": aq_scores_history,
    }

    if testing_structures is not None:
        sl_dict["mae_test_history"] = mae_test_history
        sl_dict["rmse_test_history"] = rmse_test_history
        sl_dict["test_prediction_history"] = test_pred_history
        sl_dict["test_uncertainty_history"] = test_unc_history

    if write_to_disk:
        if not os.path.isdir(write_location):
            os.makedirs(write_location)
        json_write_path = os.path.join(write_location, "sl_dict.json")
        with open(json_write_path, "w") as f:
            json.dump(sl_dict, f)
        print(f"SL dictionary written to {json_write_path}")

    return sl_dict


def choose_next_candidate(
    structures: List[Atoms] = None,
    labels: Array = None,
    train_idx: Array = None,
    pred: Array = None,
    unc: Array = None,
    aq: str = "MLI",
    num_candidates_to_pick: int = None,
    target_min: float = None,
    target_max: float = None,
    include_hhi: bool = False,
    hhi_type: str = "production",
):
    """
    Chooses the next candidate(s) from a given acquisition function

    Parameters
    ----------

    structures:
        List of `Atoms` objects to be used for HHI weighting if desired

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

    include_hhi:
        Whether HHI scores should be used to weight aq scores

    hhi_type:
        Type of HHI index to be used for weighting
        Options
        - production (default)
        - reserves

    Returns
    -------

    parent_idx:
        Index/indices of the selected candidates

    max_scores:
        Maximum scores (corresponding to the selected candidates for given `aq`)

    aq_scores:
        Calculated scores based on the selected `aq` for the entire training set
    """
    hhi_scores = None
    if include_hhi:
        if structures is None:
            msg = "Structures must be provided to include HHI scores"
            raise AutoCatSequentialLearningError(msg)
        hhi_scores = calculate_hhi_scores(structures, hhi_type)

    if aq == "Random":
        if labels is None:
            msg = "For aq = 'Random', the labels must be supplied"
            raise AutoCatSequentialLearningError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(labels), dtype=bool)

        if hhi_scores is None:
            hhi_scores = np.ones(len(train_idx))

        aq_scores = (
            np.random.choice(len(labels), size=len(labels), replace=False) * hhi_scores
        )

    elif aq == "MU":
        if unc is None:
            msg = "For aq = 'MU', the uncertainties must be supplied"
            raise AutoCatSequentialLearningError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(unc), dtype=bool)

        if hhi_scores is None:
            hhi_scores = np.ones(len(train_idx))

        aq_scores = unc.copy() * hhi_scores

    elif aq == "MLI":
        if unc is None or pred is None:
            msg = "For aq = 'MLI', both uncertainties and predictions must be supplied"
            raise AutoCatSequentialLearningError(msg)

        if train_idx is None:
            train_idx = np.zeros(len(unc), dtype=bool)

        if hhi_scores is None:
            hhi_scores = np.ones(len(train_idx))

        aq_scores = (
            np.array(
                [
                    get_overlap_score(mean, std, x2=target_max, x1=target_min)
                    for mean, std in zip(pred, unc)
                ]
            )
            * hhi_scores
        )

    else:
        msg = f"Acquisition function {aq} is not supported"
        raise NotImplementedError(msg)

    if num_candidates_to_pick is None:
        next_idx = np.array([np.argmax(aq_scores[~train_idx])])
        max_scores = [np.max(aq_scores[~train_idx])]

    else:
        next_idx = np.argsort(aq_scores[~train_idx])[-num_candidates_to_pick:]
        sorted_array = aq_scores[~train_idx][next_idx]
        max_scores = list(sorted_array[-num_candidates_to_pick:])
    parent_idx = np.arange(aq_scores.shape[0])[~train_idx][next_idx]

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


def calculate_hhi_scores(structures: List[Atoms], hhi_type: str = "production"):
    """
    Calculates HHI scores for structures weighted by their composition.
    The scores are normalized and inverted such that these should
    be maximized in the interest of finding a low cost system

    Parameters
    ----------

    structures:
        List of Atoms objects for which to calculate the scores

    hhi_type:
        Type of HHI index to be used for the score
        Options
        - production (default)
        - reserves

    Returns
    -------

    hhi_scores:
        Scores corresponding to each of the provided structures

    """
    if structures is None:
        msg = "To include HHI, the structures must be provided"
        raise AutoCatSequentialLearningError(msg)

    raw_hhi_data = {"production": HHI_PRODUCTION, "reserves": HHI_RESERVES}
    max_hhi = np.max([raw_hhi_data[hhi_type][r] for r in raw_hhi_data[hhi_type]])
    min_hhi = np.min([raw_hhi_data[hhi_type][r] for r in raw_hhi_data[hhi_type]])
    # normalize and invert (so that this score is to be maximized)
    norm_hhi_data = {
        el: 1.0 - (raw_hhi_data[hhi_type][el] - min_hhi) / (max_hhi - min_hhi)
        for el in raw_hhi_data[hhi_type]
    }

    hhi_scores = np.zeros(len(structures))
    for idx, struct in enumerate(structures):
        hhi = 0
        el_counts = struct.symbols.formula.count()
        tot_size = len(struct)
        # weight calculated hhi score by composition
        for el in el_counts:
            hhi += norm_hhi_data[el] * el_counts[el] / tot_size
        hhi_scores[idx] = hhi
    return hhi_scores
