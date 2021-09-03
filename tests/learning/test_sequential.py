"""Unit tests for the `autocat.learning.sequential` module"""

import os
import pytest
import numpy as np
import json

import tempfile

from scipy import stats

from autocat.learning.predictors import AutoCatPredictor
from autocat.learning.sequential import (
    AutoCatSequentialLearningError,
    choose_next_candidate,
    get_overlap_score,
)
from autocat.learning.sequential import simulated_sequential_learning
from autocat.learning.sequential import multiple_simulated_sequential_learning_runs
from autocat.surface import generate_surface_structures
from autocat.adsorption import place_adsorbate


def test_simulated_sequential_outputs():
    # Test outputs without any testing structures
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    base_struct2 = place_adsorbate(sub2, "NH")["custom"]["structure"]
    base_struct3 = place_adsorbate(sub2, "H")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2, base_struct3, sub1, sub2,],
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        init_training_size=1,
        batch_size_to_add=2,
        number_of_sl_loops=2,
        acquisition_function="MLI",
        target_min=0.9,
        target_max=2.1,
    )

    # Test number of train histories equals number of sl loops + 1
    assert len(sl_dict["training_history"]) == 3

    # Test initial training size
    assert len([a for a in sl_dict["training_history"][0] if a]) == 1

    # Test keeping track of history
    assert sl_dict["training_history"][0] != sl_dict["training_history"][1]

    # Test that number of max uncertainties equals number of sl loops + 1
    assert len(sl_dict["uncertainty_history"]) == 3
    # Test the number of total uncertainties collected
    assert len(sl_dict["uncertainty_history"][-1]) == 5

    # check all mae and rmse training scores collected
    assert len(sl_dict["mae_train_history"]) == 3
    assert len(sl_dict["rmse_train_history"]) == 3


def test_simulated_sequential_batch_added():
    # Tests adding N candidates on each loop
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    base_struct2 = place_adsorbate(sub2, "NH")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    bsta = 2
    num_loops = 2
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2, sub1, sub2],
        np.array([5.0, 6.0, 7.0, 8.0]),
        batch_size_to_add=bsta,
        number_of_sl_loops=num_loops,
        acquisition_function="Random",
        init_training_size=1,
    )
    assert len(sl_dict["training_history"]) == num_loops + 1
    num_in_tr_set0 = len([a for a in sl_dict["training_history"][0] if a])
    num_in_tr_set1 = len([a for a in sl_dict["training_history"][1] if a])
    num_in_tr_set2 = len([a for a in sl_dict["training_history"][2] if a])
    # should add 2 candidates on first loop
    assert num_in_tr_set1 - num_in_tr_set0 == bsta
    # since only 1 left, should add it on the next
    assert num_in_tr_set2 - num_in_tr_set1 == 1


def test_simulated_sequential_num_loops():
    # Tests the number of loops
    sub1 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "H")["custom"]["structure"]
    base_struct2 = place_adsorbate(sub2, "N")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    # Test default number of loops
    bsta = 3
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2, sub1, sub2],
        np.array([5.0, 6.0, 7.0, 8.0]),
        batch_size_to_add=bsta,
        acquisition_function="Random",
        init_training_size=1,
    )
    assert len(sl_dict["training_history"]) == 2

    # Test catches maximum number of loops
    with pytest.raises(AutoCatSequentialLearningError):
        sl_dict = simulated_sequential_learning(
            acsc,
            [base_struct1, base_struct2, sub1, sub2],
            np.array([5.0, 6.0, 7.0, 8.0]),
            batch_size_to_add=bsta,
            acquisition_function="Random",
            init_training_size=1,
            number_of_sl_loops=3,
        )

    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2, sub2],
        np.array([5.0, 6.0, 7.0]),
        acquisition_function="Random",
        init_training_size=1,
    )
    assert len(sl_dict["training_history"]) == 3


def test_simulated_sequential_outputs_testing():
    # Test with testing structures given
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    base_struct2 = place_adsorbate(sub2, "NH")["custom"]["structure"]
    base_struct3 = place_adsorbate(sub2, "H")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2, sub1],
        np.array([0.0, 2.0, 4.0]),
        init_training_size=1,
        testing_structures=[base_struct3, sub2],
        testing_y=np.array([8.0, 10.0]),
        batch_size_to_add=1,
        number_of_sl_loops=2,
        acquisition_function="MU",
    )
    # Check length of testing scores
    assert len(sl_dict["training_history"]) == 3
    assert len(sl_dict["aq_scores_history"]) == 2
    assert len(sl_dict["max_scores_history"]) == 2
    assert len(sl_dict["mae_test_history"]) == 3
    assert len(sl_dict["rmse_train_history"]) == 3
    assert len(sl_dict["mae_train_history"]) == 3
    assert len(sl_dict["test_prediction_history"]) == 3
    assert len(sl_dict["test_prediction_history"][0]) == 2
    assert len(sl_dict["test_uncertainty_history"]) == 3
    assert len(sl_dict["test_uncertainty_history"][0]) == 2
    assert sl_dict["mae_test_history"] != sl_dict["mae_train_history"]


def test_simulated_sequential_write_to_disk():
    # Test writing out sl dict
    _tmp_dir = tempfile.TemporaryDirectory().name
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    base_struct2 = place_adsorbate(sub2, "NH")["custom"]["structure"]
    base_struct3 = place_adsorbate(sub2, "N")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2, base_struct3],
        np.array([0, 1, 2]),
        init_training_size=2,
        testing_structures=[base_struct3],
        testing_y=np.array([2]),
        batch_size_to_add=2,
        number_of_sl_loops=1,
        write_to_disk=True,
        write_location=_tmp_dir,
        acquisition_function="Random",
    )
    # check data written as json
    with open(os.path.join(_tmp_dir, "sl_dict.json"), "r") as f:
        sl_written = json.load(f)
        assert sl_dict == sl_written


def test_multiple_sequential_learning_serial():
    # Tests serial implementation
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    runs_history = multiple_simulated_sequential_learning_runs(
        number_of_runs=3,
        sl_kwargs={
            "predictor": acsc,
            "all_training_structures": [base_struct1, sub1],
            "all_training_y": np.array([0.0, 0.0]),
            "number_of_sl_loops": 1,
            "acquisition_function": "MU",
            "init_training_size": 1,
        },
    )
    assert len(runs_history) == 3
    assert isinstance(runs_history[0], dict)
    assert "mae_train_history" in runs_history[1]


def test_multiple_sequential_learning_parallel():
    # Tests parallel implementation
    sub1 = generate_surface_structures(["Cu"], facets={"Cu": ["111"]})["Cu"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "Li")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    runs_history = multiple_simulated_sequential_learning_runs(
        number_of_runs=3,
        number_parallel_jobs=2,
        sl_kwargs={
            "predictor": acsc,
            "all_training_structures": [base_struct1, sub1],
            "all_training_y": np.array([0.0, 0.0]),
            "number_of_sl_loops": 1,
            "acquisition_function": "Random",
            "init_training_size": 1,
        },
    )
    assert len(runs_history) == 3
    assert isinstance(runs_history[2], dict)
    assert "rmse_train_history" in runs_history[0]


def test_multiple_sequential_learning_write_to_disk():
    # Tests writing run history to disk
    _tmp_dir = tempfile.TemporaryDirectory().name
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "N")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    runs_history = multiple_simulated_sequential_learning_runs(
        number_of_runs=3,
        number_parallel_jobs=2,
        write_to_disk=True,
        write_location=_tmp_dir,
        sl_kwargs={
            "predictor": acsc,
            "all_training_structures": [base_struct1, sub1],
            "all_training_y": np.array([0.0, 0.0]),
            "number_of_sl_loops": 1,
            "acquisition_function": "Random",
            "init_training_size": 1,
            "batch_size_to_add": 2,
        },
    )

    # check data history
    with open(os.path.join(_tmp_dir, "sl_runs_history.json"), "r") as f:
        runs_history_written = json.load(f)
        assert runs_history == runs_history_written


def test_choose_next_candidate_input_minimums():
    # Tests that appropriately catches minimum necessary inputs
    labels = np.random.rand(5)
    train_idx = np.zeros(5, dtype=bool)
    train_idx[np.random.choice(5, size=2, replace=False)] = 1
    unc = np.random.rand(5)
    pred = np.random.rand(5)

    with pytest.raises(AutoCatSequentialLearningError):
        choose_next_candidate()

    with pytest.raises(AutoCatSequentialLearningError):
        choose_next_candidate(unc=unc, pred=pred, num_candidates_to_pick=2, aq="Random")

    with pytest.raises(AutoCatSequentialLearningError):
        choose_next_candidate(
            labels=labels, pred=pred, num_candidates_to_pick=2, aq="MU"
        )

    with pytest.raises(AutoCatSequentialLearningError):
        choose_next_candidate(pred=pred, num_candidates_to_pick=2, aq="MLI")

    with pytest.raises(AutoCatSequentialLearningError):
        choose_next_candidate(unc=unc, num_candidates_to_pick=2, aq="MLI")


def test_get_overlap_score():
    # Tests default behavior
    mean = 0.0
    std = 0.1
    x1 = -0.4
    x2 = 0.8
    norm = stats.norm(loc=mean, scale=std)

    # checks that at least target min or max is provided
    with pytest.raises(AutoCatSequentialLearningError):
        get_overlap_score(mean, std)

    # test default min
    overlap_score = get_overlap_score(mean, std, x2=x2)
    assert np.isclose(overlap_score, norm.cdf(x2))

    # test default max
    overlap_score = get_overlap_score(mean, std, x1=x1)
    assert np.isclose(overlap_score, 1.0 - norm.cdf(x1))

    # test both max and min
    overlap_score = get_overlap_score(mean, std, x1=x1, x2=x2)
    assert np.isclose(overlap_score, norm.cdf(x2) - norm.cdf(x1))
