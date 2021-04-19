"""Unit tests for the `autocat.learning.sequential` module"""

import os
import pytest
import numpy as np
import json

import tempfile

from autocat.learning.predictors import AutoCatStructureCorrector
from autocat.learning.sequential import simulated_sequential_learning
from autocat.learning.sequential import multiple_sequential_learning_runs
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
    acsc = AutoCatStructureCorrector(structure_featurizer="elemental_property")
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2],
        initial_num_of_perturbations_per_base_structure=4,
        batch_num_of_perturbations_per_base_structure=3,
        number_of_sl_loops=4,
    )
    # Test that number of max uncertainties equals number of sl loops
    assert len(sl_dict["max_unc_history"]) == 4
    # Test the number of total uncertainties collected
    assert len(sl_dict["full_unc_history"][-1]) == 6
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2],
        batch_num_of_perturbations_per_base_structure=1,
        number_of_sl_loops=4,
    )
    # check default for initial number of training perturbations
    assert len(sl_dict["full_unc_history"][0]) == 2

    # check all mae and rmse training scores collected
    assert len(sl_dict["mae_train_history"]) == 4
    assert len(sl_dict["rmse_train_history"]) == 4

    # check prediction and correction history
    assert len(sl_dict["pred_corrs_history"]) == len(sl_dict["real_corrs_history"])
    assert len(sl_dict["pred_corrs_history"][-1]) == len(
        sl_dict["real_corrs_history"][-1]
    )
    assert len(sl_dict["pred_corrs_history"]) == 4
    assert len(sl_dict["real_corrs_history"]) == 4


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
    acsc = AutoCatStructureCorrector(structure_featurizer="elemental_property")
    bsta = 2
    num_loops = 3
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2],
        batch_num_of_perturbations_per_base_structure=4,
        batch_size_to_add=bsta,
        number_of_sl_loops=num_loops,
    )
    # each max unc history should be equal to number of candidates added
    assert len(sl_dict["max_unc_history"][0]) == bsta
    # first dimension is number of loops
    assert len(sl_dict["pred_corrs_history"]) == num_loops
    # next dimension is size of candidates added on each loop
    assert len(sl_dict["pred_corrs_history"][0]) == bsta
    # next dimension is size of adsorbate
    assert len(sl_dict["pred_corrs_history"][0][0]) == 2
    # next dimension is vector correction to atom in adsorbate
    assert len(sl_dict["pred_corrs_history"][0][0][0]) == 3
    # check same holds for real history
    assert len(sl_dict["real_corrs_history"]) == num_loops
    assert len(sl_dict["real_corrs_history"][0]) == bsta
    assert len(sl_dict["real_corrs_history"][0][0]) == 2
    assert len(sl_dict["real_corrs_history"][0][0][0]) == 3


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
    acsc = AutoCatStructureCorrector(structure_featurizer="elemental_property")
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2],
        testing_base_structures=[base_struct3],
        batch_num_of_perturbations_per_base_structure=3,
        number_of_sl_loops=2,
    )
    # Check lenght of testing scores
    assert len(sl_dict["mae_test_history"]) == 2
    assert len(sl_dict["rmse_train_history"]) == 2
    assert len(sl_dict["mae_train_history"]) == 2
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
    acsc = AutoCatStructureCorrector(structure_featurizer="elemental_property")
    sl_dict = simulated_sequential_learning(
        acsc,
        [base_struct1, base_struct2],
        testing_base_structures=[base_struct3],
        batch_num_of_perturbations_per_base_structure=1,
        number_of_sl_loops=1,
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    with open(os.path.join(_tmp_dir, "sl_dict.json"), "r") as f:
        sl_written = json.load(f)
        assert sl_dict == sl_written


def test_multiple_sequential_learning_serial():
    # Tests serial implementation
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    acsc = AutoCatStructureCorrector(structure_featurizer="elemental_property")
    runs_history = multiple_sequential_learning_runs(
        acsc,
        [base_struct1],
        3,
        batch_num_of_perturbations_per_base_structure=1,
        number_of_sl_loops=2,
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
    acsc = AutoCatStructureCorrector(structure_featurizer="elemental_property")
    runs_history = multiple_sequential_learning_runs(
        acsc,
        [base_struct1],
        3,
        batch_num_of_perturbations_per_base_structure=1,
        number_of_sl_loops=2,
        number_parallel_jobs=2,
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
    acsc = AutoCatStructureCorrector(structure_featurizer="elemental_property")
    runs_history = multiple_sequential_learning_runs(
        acsc,
        [base_struct1],
        3,
        batch_num_of_perturbations_per_base_structure=1,
        number_of_sl_loops=2,
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    with open(os.path.join(_tmp_dir, "sl_runs_history.json"), "r") as f:
        runs_history_written = json.load(f)
        assert runs_history == runs_history_written
