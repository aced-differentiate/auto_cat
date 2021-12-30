"""Unit tests for the `autocat.learning.sequential` module"""

import os
import pytest
import numpy as np
import json

import tempfile

from scipy import stats
from ase.io import read as ase_read
from autocat.data.hhi import HHI_PRODUCTION
from autocat.data.hhi import HHI_RESERVES
from autocat.data.segregation_energies import SEGREGATION_ENERGIES
from autocat.learning.predictors import AutoCatPredictor
from autocat.learning.sequential import (
    AutoCatDesignSpace,
    AutoCatDesignSpaceError,
    AutoCatSequentialLearningError,
    AutoCatSequentialLearner,
    calculate_segregation_energy_scores,
    choose_next_candidate,
    get_overlap_score,
)
from autocat.learning.sequential import simulated_sequential_learning
from autocat.learning.sequential import multiple_simulated_sequential_learning_runs
from autocat.learning.sequential import calculate_hhi_scores
from autocat.surface import generate_surface_structures
from autocat.adsorption import place_adsorbate
from autocat.saa import generate_saa_structures
from autocat.utils import extract_structures


def test_sequential_learner_from_json():
    # Tests generation of an AutoCatSequentialLearner from a json
    sub1 = generate_surface_structures(["Au"], facets={"Au": ["110"]})["Au"]["fcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, "C")["custom"]["structure"]
    sub2 = generate_surface_structures(["Li"], facets={"Li": ["100"]})["Li"]["bcc100"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, "Mg")["custom"]["structure"]
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, "N")["custom"]["structure"]
    structs = [sub1, sub2, sub3]
    labels = np.array([0.1, 0.2, 0.3])
    predictor_kwargs = {
        "structure_featurizer": "coulomb_matrix",
        "elementalproperty_preset": "megnet_el",
        "adsorbate_featurizer": "soap",
        "adsorbate_featurization_kwargs": {"rcut": 5.0, "nmax": 8, "lmax": 6},
        "species_list": ["Au", "Li", "Mg", "C", "Ru", "N"],
    }

    candidate_selection_kwargs = {"aq": "Random", "num_candidates_to_pick": 3}
    acds = AutoCatDesignSpace(structs, labels)
    acsl = AutoCatSequentialLearner(
        acds,
        predictor_kwargs=predictor_kwargs,
        candidate_selection_kwargs=candidate_selection_kwargs,
    )
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsl.write_json(_tmp_dir, "testing_acsl.json")
        json_path = os.path.join(_tmp_dir, "testing_acsl.json")
        written_acsl = AutoCatSequentialLearner.from_json(json_path)
        assert np.array_equal(
            written_acsl.design_space.design_space_labels, acds.design_space_labels
        )
        assert (
            written_acsl.design_space.design_space_structures
            == acds.design_space_structures
        )
        assert written_acsl.predictor_kwargs == predictor_kwargs
        assert written_acsl.candidate_selection_kwargs == candidate_selection_kwargs


def test_sequential_learner_write_json():
    # Tests writing a AutoCatSequentialLearner to disk as a json
    sub1 = generate_surface_structures(["Ag"], facets={"Ag": ["110"]})["Ag"]["fcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, "B")["custom"]["structure"]
    sub2 = generate_surface_structures(["Li"], facets={"Li": ["100"]})["Li"]["bcc100"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, "Al")["custom"]["structure"]
    sub3 = generate_surface_structures(["Ti"], facets={"Ti": ["0001"]})["Ti"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, "H")["custom"]["structure"]
    structs = [sub1, sub2, sub3]
    labels = np.array([0.1, 0.2, 0.3])
    predictor_kwargs = {
        "structure_featurizer": "sine_matrix",
        "elementalproperty_preset": "deml",
        "adsorbate_featurizer": "soap",
        "adsorbate_featurization_kwargs": {"rcut": 5.0, "nmax": 8, "lmax": 6},
        "species_list": ["Ag", "Li", "Al", "B", "Ti", "H"],
    }

    candidate_selection_kwargs = {"aq": "MU", "num_candidates_to_pick": 2}
    acds = AutoCatDesignSpace(structs, labels)
    acsl = AutoCatSequentialLearner(
        acds,
        predictor_kwargs=predictor_kwargs,
        candidate_selection_kwargs=candidate_selection_kwargs,
    )
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsl.write_json(_tmp_dir, "testing_acsl.json")
        with open(os.path.join(_tmp_dir, "testing_acsl.json"), "r") as f:
            sl = json.load(f)
        # collects structs by writing each json individually
        # and reading with ase
        written_structs = []
        for i in range(3):
            _tmp_json = os.path.join(_tmp_dir, "tmp.json")
            with open(_tmp_json, "w") as tmp:
                json.dump(sl[i], tmp)
            written_structs.append(ase_read(_tmp_json))
        assert structs == written_structs
        assert (labels == sl[3]).all()
        # check predictor kwargs kept
        assert predictor_kwargs == sl[4]
        # check candidate selection kwargs kept
        assert candidate_selection_kwargs == sl[-2]
        assert sl[-1] == {
            "iteration_count": 0,
            "train_idx": None,
            "predictions": None,
            "predictions_history": None,
            "uncertainties": None,
            "uncertainties_history": None,
            "candidate_indices": None,
            "candidate_index_history": None,
            "aq_scores": None,
        }

    # test writing when no kwargs given
    acsl = AutoCatSequentialLearner(acds)
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsl.write_json(_tmp_dir)
        with open(os.path.join(_tmp_dir, "acsl.json"), "r") as f:
            sl = json.load(f)
        # collects structs by writing each json individually
        # and reading with ase
        written_structs = []
        for i in range(3):
            _tmp_json = os.path.join(_tmp_dir, "tmp.json")
            with open(_tmp_json, "w") as tmp:
                json.dump(sl[i], tmp)
            written_structs.append(ase_read(_tmp_json))
        assert structs == written_structs
        assert (labels == sl[3]).all()
        # check default predictor kwargs kept
        assert sl[4] == {"structure_featurizer": "sine_matrix"}
        # check default candidate selection kwargs kept
        assert sl[-2] == {"aq": "Random"}


def test_sequential_learner_iterate():
    # Tests iterate method
    sub1 = generate_surface_structures(["Ca"], facets={"Ca": ["111"]})["Ca"]["fcc111"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, "Na")["custom"]["structure"]
    sub2 = generate_surface_structures(["Nb"], facets={"Nb": ["110"]})["Nb"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, "K")["custom"]["structure"]
    sub3 = generate_surface_structures(["Ta"], facets={"Ta": ["110"]})["Ta"]["bcc110"][
        "structure"
    ]
    sub3 = place_adsorbate(sub3, "H")["custom"]["structure"]
    sub4 = generate_surface_structures(["Sr"], facets={"Sr": ["110"]})["Sr"]["fcc110"][
        "structure"
    ]
    sub4 = place_adsorbate(sub4, "Fe")["custom"]["structure"]
    structs = [sub1, sub2, sub3, sub4]
    labels = np.array([11.0, 25.0, np.nan, np.nan])
    acds = AutoCatDesignSpace(structs, labels)
    acsl = AutoCatSequentialLearner(acds)

    assert acsl.iteration_count == 0

    acsl.iterate()
    assert acsl.iteration_count == 1
    assert acsl.predictions is not None
    assert len(acsl.predictions_history) == 1
    assert len(acsl.predictions_history[0]) == len(acds)
    assert acsl.uncertainties is not None
    assert len(acsl.uncertainties_history) == 1
    assert len(acsl.uncertainties_history[0]) == len(acds)
    assert acsl.candidate_indices is not None
    assert acsl.candidate_index_history is not None
    assert acsl.candidate_index_history == [acsl.candidate_indices]

    cand_ind1 = acsl.candidate_indices[0]
    acsl.design_space.update([structs[cand_ind1]], np.array([13.0]))

    acsl.iterate()
    assert acsl.iteration_count == 2

    # checks being iterated a second time to fully explore the design space
    cand_ind2 = acsl.candidate_indices[0]
    assert cand_ind1 != cand_ind2
    assert acsl.candidate_index_history == [[cand_ind1], [cand_ind2]]
    assert len(acsl.uncertainties_history) == 2
    assert len(acsl.predictions_history) == 2

    acsl.design_space.update([structs[cand_ind2]], np.array([17.0]))
    acsl.iterate()

    assert acsl.iteration_count == 3
    assert acsl.candidate_structures is None
    assert acsl.candidate_indices is None
    assert acsl.candidate_index_history == [[cand_ind1], [cand_ind2]]
    assert len(acsl.uncertainties_history) == 3
    assert len(acsl.predictions_history) == 3


def test_sequential_learner_setup():
    # Tests setting up an SL object
    sub1 = generate_surface_structures(["Ir"], facets={"Ir": ["100"]})["Ir"]["fcc100"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    sub2 = generate_surface_structures(["Mo"], facets={"Mo": ["110"]})["Mo"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, "H")["custom"]["structure"]
    sub3 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub3 = place_adsorbate(sub3, "O")["custom"]["structure"]
    sub4 = generate_surface_structures(["Re"], facets={"Re": ["0001"]})["Re"][
        "hcp0001"
    ]["structure"]
    sub4 = place_adsorbate(sub4, "N")["custom"]["structure"]
    structs = [sub1, sub2, sub3, sub4]
    labels = np.array([4.0, np.nan, 6.0, np.nan])
    acds = AutoCatDesignSpace(structs, labels)
    acsl = AutoCatSequentialLearner(acds)

    assert acsl.design_space == acds
    assert acsl.iteration_count == 0
    assert acsl.predictions == None
    assert acsl.candidate_indices == None
    assert acsl.candidate_selection_kwargs == {"aq": "Random"}
    # test default kwargs
    assert acsl.predictor_kwargs == {"structure_featurizer": "sine_matrix"}
    # test specifying kwargs
    acsl = AutoCatSequentialLearner(
        acds,
        predictor_kwargs={
            "structure_featurizer": "elemental_property",
            "elementalproperty_preset": "pymatgen",
            "adsorbate_featurizer": "soap",
            "adsorbate_featurization_kwargs": {"rcut": 5.0, "nmax": 8, "lmax": 6},
            "species_list": ["Ir", "O", "H", "Mo", "Fe", "N", "Re"],
            "model_kwargs": {"n_restarts_optimizer": 9},
        },
        candidate_selection_kwargs={"aq": "MU", "num_candidates_to_pick": 2},
    )
    # test passing predictor kwargs
    assert acsl.predictor_kwargs == {
        "structure_featurizer": "elemental_property",
        "elementalproperty_preset": "pymatgen",
        "adsorbate_featurizer": "soap",
        "adsorbate_featurization_kwargs": {"rcut": 5.0, "nmax": 8, "lmax": 6},
        "species_list": ["Ir", "O", "H", "Mo", "Fe", "N", "Re"],
        "model_kwargs": {"n_restarts_optimizer": 9},
    }
    assert acsl.predictor.structure_featurizer == "elemental_property"
    assert acsl.predictor.elementalproperty_preset == "pymatgen"
    assert acsl.predictor.adsorbate_featurizer == "soap"
    assert acsl.predictor.adsorbate_featurization_kwargs == {
        "rcut": 5.0,
        "nmax": 8,
        "lmax": 6,
    }

    # test passing candidate selection kwargs
    assert acsl.candidate_selection_kwargs == {"aq": "MU", "num_candidates_to_pick": 2}


def test_design_space_setup():
    # test setting up an AutoCatDesignSpace
    sub1 = generate_surface_structures(
        ["Pt"], supercell_dim=[2, 2, 5], facets={"Pt": ["100"]}
    )["Pt"]["fcc100"]["structure"]
    sub1 = place_adsorbate(sub1, "H2")["custom"]["structure"]
    sub2 = generate_surface_structures(["Na"], facets={"Na": ["110"]})["Na"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, "F")["custom"]["structure"]
    structs = [sub1, sub2]
    labels = np.array([3.0, 7.0])
    acds = AutoCatDesignSpace(structs, labels)
    assert acds.design_space_structures == [sub1, sub2]
    assert acds.design_space_structures is not structs
    assert np.array_equal(acds.design_space_labels, labels)
    assert acds.design_space_labels is not labels
    assert len(acds) == 2
    # test different number of structures and labels
    with pytest.raises(AutoCatDesignSpaceError):
        acds = AutoCatDesignSpace([sub1], labels)


def test_delitem_design_space():
    # tests deleting items from the design space
    sub0 = generate_surface_structures(["Pd"], facets={"Pd": ["100"]})["Pd"]["fcc100"][
        "structure"
    ]
    sub0 = place_adsorbate(sub0, "O")["custom"]["structure"]
    sub1 = generate_surface_structures(["V"], facets={"V": ["110"]})["V"]["bcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, "H")["custom"]["structure"]
    sub2 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, "S")["custom"]["structure"]
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, "P")["custom"]["structure"]
    structs = [sub0, sub1, sub2]
    labels = np.array([-2.5, np.nan, 600.0])
    # test deleting by single idx
    acds = AutoCatDesignSpace(structs, labels)
    del acds[1]
    assert len(acds) == 2
    assert np.array_equal(acds.design_space_labels, np.array([-2.5, 600.0]))
    assert acds.design_space_structures == [sub0, sub2]
    # test deleting using a mask
    acds = AutoCatDesignSpace(structs, labels)
    mask = np.zeros(len(acds), bool)
    mask[0] = 1
    mask[1] = 1
    # n.b. deletes wherever mask is True
    del acds[mask]
    assert len(acds) == 1
    assert acds.design_space_structures == [sub2]
    assert np.array_equal(acds.design_space_labels, np.array([600.0]))
    # test deleting by providing list of idx
    structs = [sub0, sub1, sub2, sub3]
    labels = np.array([-20, 8, np.nan, 0.3])
    acds = AutoCatDesignSpace(structs, labels)
    del acds[[1, 3]]
    assert len(acds) == 2
    assert np.array_equal(
        acds.design_space_labels, np.array([-20, np.nan]), equal_nan=True
    )
    assert acds.design_space_structures == [sub0, sub2]
    # test deleting by providing list with a single idx
    acds = AutoCatDesignSpace(structs, labels)
    del acds[[0]]
    assert len(acds) == 3
    assert np.array_equal(
        acds._design_space_labels, np.array([8, np.nan, 0.3]), equal_nan=True
    )
    assert acds.design_space_structures == [sub1, sub2, sub3]


def test_updating_design_space():
    sub1 = generate_surface_structures(["Ag"], facets={"Ag": ["100"]})["Ag"]["fcc100"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Li"], facets={"Li": ["110"]})["Li"]["bcc110"][
        "structure"
    ]
    sub3 = generate_surface_structures(["Na"], facets={"Na": ["110"]})["Na"]["bcc110"][
        "structure"
    ]
    sub4 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    structs = [sub1, sub2, sub3]
    labels = np.array([4.0, 5.0, 6.0])
    acds = AutoCatDesignSpace(structs, labels)

    # Test trying to update just structures
    with pytest.raises(AutoCatDesignSpaceError):
        acds.design_space_structures = [sub4]

    # Test trying to update just labels
    with pytest.raises(AutoCatDesignSpaceError):
        acds.design_space_structures = np.array([4.0])

    # Test updating label already in DS and extending
    acds.update([sub1, sub4], np.array([10.0, 20.0]))
    assert np.isclose(acds.design_space_labels[0], 10.0)
    assert sub4 in acds.design_space_structures
    assert np.isclose(acds.design_space_labels[-1], 20.0)

    # Test trying to give structures that are not Atoms objects
    with pytest.raises(AssertionError):
        acds.update([sub1, np.array(20.0)], np.array([3.0, 4.0]))


def test_write_design_space_as_json():
    # Tests writing out the AutoCatDesignSpace to disk
    sub1 = generate_surface_structures(["Pd"], facets={"Pd": ["111"]})["Pd"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["V"], facets={"V": ["110"]})["V"]["bcc110"][
        "structure"
    ]
    structs = [sub1, sub2]
    labels = np.array([0.3, 0.8])
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acds = AutoCatDesignSpace(
            design_space_structures=structs, design_space_labels=labels,
        )
        acds.write_json(write_location=_tmp_dir)
        # loads back written json
        with open(os.path.join(_tmp_dir, "acds.json"), "r") as f:
            ds = json.load(f)
        # collects structs by writing each json individually
        # and reading with ase
        written_structs = []
        for i in range(2):
            _tmp_json = os.path.join(_tmp_dir, "tmp.json")
            with open(_tmp_json, "w") as tmp:
                json.dump(ds[i], tmp)
            written_structs.append(ase_read(_tmp_json))
        assert structs == written_structs
        assert (labels == ds[-1]).all()


def test_get_design_space_from_json():
    # Tests generating AutoCatDesignSpace from a json
    sub1 = generate_surface_structures(["Au"], facets={"Au": ["100"]})["Au"]["fcc100"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    structs = [sub1, sub2, sub3]
    labels = np.array([30.0, 900.0, np.nan])
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acds = AutoCatDesignSpace(
            design_space_structures=structs, design_space_labels=labels,
        )
        acds.write_json("testing.json", write_location=_tmp_dir)

        tmp_json_dir = os.path.join(_tmp_dir, "testing.json")
        acds_from_json = AutoCatDesignSpace.from_json(tmp_json_dir)
        assert acds_from_json.design_space_structures == structs
        assert np.array_equal(
            acds_from_json.design_space_labels, labels, equal_nan=True
        )


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
    ds_structs = [
        base_struct1,
        base_struct2,
        base_struct3,
        sub1,
        sub2,
    ]
    ds_labels = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    sl_dict = simulated_sequential_learning(
        predictor=acsc,
        design_space=acds,
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
    ds_structs = [base_struct1, base_struct2, sub1, sub2]
    ds_labels = np.array([5.0, 6.0, 7.0, 8.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    sl_dict = simulated_sequential_learning(
        predictor=acsc,
        design_space=acds,
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
    ds_structs = [base_struct1, base_struct2, sub1, sub2]
    ds_labels = np.array([5.0, 6.0, 7.0, 8.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    # Test default number of loops
    bsta = 3
    sl_dict = simulated_sequential_learning(
        predictor=acsc,
        design_space=acds,
        batch_size_to_add=bsta,
        acquisition_function="Random",
        init_training_size=1,
    )
    assert len(sl_dict["training_history"]) == 2

    # Test catches maximum number of loops
    with pytest.raises(AutoCatSequentialLearningError):
        sl_dict = simulated_sequential_learning(
            predictor=acsc,
            design_space=acds,
            batch_size_to_add=bsta,
            acquisition_function="Random",
            init_training_size=1,
            number_of_sl_loops=3,
        )

    ds_structs = [base_struct1, base_struct2, sub2]
    ds_labels = np.array([5.0, 6.0, 7.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)

    sl_dict = simulated_sequential_learning(
        predictor=acsc,
        design_space=acds,
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
    ds_structs = [base_struct1, base_struct2, sub1]
    ds_labels = np.array([0.0, 2.0, 4.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    sl_dict = simulated_sequential_learning(
        predictor=acsc,
        design_space=acds,
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
    ds_structs = [base_struct1, base_struct2, base_struct3]
    ds_labels = np.array([0, 1, 2])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    sl_dict = simulated_sequential_learning(
        predictor=acsc,
        design_space=acds,
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


def test_simulated_sequential_learning_fully_explored():
    # Checks that catches if ds not fully explored
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
    ds_structs = [base_struct1, base_struct2, sub2]
    ds_labels = np.array([0.0, np.nan, 4.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    with pytest.raises(AutoCatSequentialLearningError):
        sl_dict = simulated_sequential_learning(
            predictor=acsc,
            design_space=acds,
            init_training_size=1,
            testing_structures=[base_struct3, sub2],
            testing_y=np.array([8.0, 10.0]),
            batch_size_to_add=1,
            number_of_sl_loops=2,
            acquisition_function="MU",
        )


def test_multiple_sequential_learning_serial():
    # Tests serial implementation
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    acsc = AutoCatPredictor(structure_featurizer="elemental_property")
    ds_structs = [base_struct1, sub1]
    ds_labels = np.array([0.0, 0.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    runs_history = multiple_simulated_sequential_learning_runs(
        number_of_runs=3,
        sl_kwargs={
            "predictor": acsc,
            "design_space": acds,
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
    ds_structs = [base_struct1, sub1]
    ds_labels = np.array([0.0, 0.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    runs_history = multiple_simulated_sequential_learning_runs(
        number_of_runs=3,
        number_parallel_jobs=2,
        sl_kwargs={
            "predictor": acsc,
            "design_space": acds,
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
    ds_structs = [base_struct1, sub1]
    ds_labels = np.array([0.0, 0.0])
    acds = AutoCatDesignSpace(ds_structs, ds_labels)
    runs_history = multiple_simulated_sequential_learning_runs(
        number_of_runs=3,
        number_parallel_jobs=2,
        write_to_disk=True,
        write_location=_tmp_dir,
        sl_kwargs={
            "predictor": acsc,
            "design_space": acds,
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


def test_choose_next_candidate_hhi_weighting():
    # Tests that the HHI weighting is properly applied
    unc = np.array([0.1, 0.1])
    pred = np.array([4.0, 4.0])
    # Tests using production HHI values and MU
    y_struct = generate_surface_structures(["Y"], facets={"Y": ["0001"]})["Y"][
        "hcp0001"
    ]["structure"]
    ni_struct = generate_surface_structures(["Ni"], facets={"Ni": ["111"]})["Ni"][
        "fcc111"
    ]["structure"]
    parent_idx, _, aq_scores = choose_next_candidate(
        [y_struct, ni_struct],
        unc=unc,
        include_hhi=True,
        aq="MU",
        include_seg_ener=False,
    )
    assert parent_idx[0] == 1
    assert aq_scores[0] < aq_scores[1]

    # Tests using reserves HHI values and MLI
    nb_struct = generate_surface_structures(["Nb"], facets={"Nb": ["111"]})["Nb"][
        "bcc111"
    ]["structure"]
    na_struct = generate_surface_structures(["Na"], facets={"Na": ["110"]})["Na"][
        "bcc110"
    ]["structure"]
    parent_idx, _, aq_scores = choose_next_candidate(
        [na_struct, nb_struct],
        unc=unc,
        pred=pred,
        target_min=3,
        target_max=5,
        include_hhi=True,
        hhi_type="reserves",
        include_seg_ener=False,
    )
    assert parent_idx[0] == 0
    assert aq_scores[0] > aq_scores[1]


def test_choose_next_candidate_segregation_energy_weighting():
    # Tests that the segregation energy weighting is properly applied
    unc = np.array([0.3, 0.3])
    pred = np.array([2.0, 2.0])
    structs = extract_structures(
        generate_saa_structures(["Cr"], ["Rh"], facets={"Cr": ["110"]})
    )
    structs.extend(
        extract_structures(
            generate_saa_structures(["Co"], ["Re"], facets={"Co": ["0001"]})
        )
    )
    parent_idx, _, aq_scores = choose_next_candidate(
        structs,
        unc=unc,
        pred=pred,
        target_min=0,
        target_max=4,
        include_hhi=False,
        include_seg_ener=True,
    )
    assert parent_idx[0] == 0
    assert aq_scores[0] > aq_scores[1]


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


def test_calculate_hhi_scores():
    # Tests calculating the HHI scores
    saa_dict = generate_saa_structures(
        ["Pt", "Cu", "Ni"],
        ["Ru"],
        facets={"Pt": ["111"], "Cu": ["111"], "Ni": ["111"]},
    )
    saa_structs = [saa_dict[host]["Ru"]["fcc111"]["structure"] for host in saa_dict]
    # test production
    hhi_prod_scores = calculate_hhi_scores(saa_structs)
    norm_hhi_prod = {
        el: 1.0 - (HHI_PRODUCTION[el] - 500.0) / 9300.0 for el in HHI_PRODUCTION
    }
    # check approach properly normalizes and inverts
    assert np.isclose(norm_hhi_prod["Y"], 0.0)
    assert np.isclose(norm_hhi_prod["O"], 1.0)
    # test scores calculated on SAAs
    assert np.isclose(
        hhi_prod_scores[0], (35 * norm_hhi_prod["Pt"] + norm_hhi_prod["Ru"]) / 36
    )
    assert np.isclose(
        hhi_prod_scores[1], (35 * norm_hhi_prod["Cu"] + norm_hhi_prod["Ru"]) / 36
    )
    assert np.isclose(
        hhi_prod_scores[2], (35 * norm_hhi_prod["Ni"] + norm_hhi_prod["Ru"]) / 36
    )
    # check scores normalized
    assert (hhi_prod_scores <= 1.0).all()
    assert (hhi_prod_scores >= 0.0).all()
    # test reserves
    hhi_res_scores = calculate_hhi_scores(saa_structs, "reserves")
    norm_hhi_res = {
        el: 1.0 - (HHI_RESERVES[el] - 500.0) / 8600.0 for el in HHI_RESERVES
    }
    # check approach properly normalizes and inverts
    assert np.isclose(norm_hhi_res["Pt"], 0.0)
    assert np.isclose(norm_hhi_res["C"], 1.0)
    assert np.isclose(
        hhi_res_scores[0], (35 * norm_hhi_res["Pt"] + norm_hhi_res["Ru"]) / 36
    )
    assert np.isclose(
        hhi_res_scores[1], (35 * norm_hhi_res["Cu"] + norm_hhi_res["Ru"]) / 36
    )
    assert np.isclose(
        hhi_res_scores[2], (35 * norm_hhi_res["Ni"] + norm_hhi_res["Ru"]) / 36
    )
    # check normalized
    assert (hhi_res_scores <= 1.0).all()
    assert (hhi_res_scores >= 0.0).all()


def test_calculate_segregation_energy_scores():
    # Tests calculating segregation energy scores
    saa_structs = extract_structures(
        generate_saa_structures(
            ["Ag", "Ni"], ["Pt"], facets={"Ag": ["111"], "Ni": ["111"]},
        )
    )
    saa_structs.extend(
        extract_structures(
            generate_saa_structures(["Pd"], ["W"], facets={"Pd": ["111"]})
        )
    )
    # saa_structs = [saa_dict[host]["Pt"]["fcc111"]["structure"] for host in saa_dict]
    se_scores = calculate_segregation_energy_scores(saa_structs)
    assert np.isclose(se_scores[-1], 0.0)
    min_seg = SEGREGATION_ENERGIES["Fe_100"]["Ag"]
    max_seg = SEGREGATION_ENERGIES["Pd"]["W"]
    assert np.isclose(
        se_scores[0],
        1.0 - (SEGREGATION_ENERGIES["Ag"]["Pt"] - min_seg) / (max_seg - min_seg),
    )
    assert np.isclose(
        se_scores[1],
        1.0 - (SEGREGATION_ENERGIES["Ni"]["Pt"] - min_seg) / (max_seg - min_seg),
    )
