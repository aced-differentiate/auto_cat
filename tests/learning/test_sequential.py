"""Unit tests for the `autocat.learning.sequential` module"""

import os
import pytest
import numpy as np
import json

import tempfile
from sklearn.ensemble import RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from dscribe.descriptors import SOAP
from dscribe.descriptors import SineMatrix
from matminer.featurizers.composition import ElementProperty

from scipy import stats
from ase.io.jsonio import decode as ase_decoder
from ase.io.jsonio import encode as atoms_encoder
from ase import Atoms
from autocat.data.hhi import HHI
from autocat.data.segregation_energies import SEGREGATION_ENERGIES
from autocat.learning.featurizers import Featurizer
from autocat.learning.predictors import Predictor
from autocat.learning.sequential import (
    DesignSpace,
    DesignSpaceError,
    SequentialLearnerError,
    SequentialLearner,
    calculate_segregation_energy_scores,
    get_overlap_score,
)
from autocat.learning.sequential import CandidateSelector
from autocat.learning.sequential import CandidateSelectorError
from autocat.learning.sequential import simulated_sequential_learning
from autocat.learning.sequential import multiple_simulated_sequential_learning_runs
from autocat.learning.sequential import calculate_hhi_scores
from autocat.surface import generate_surface_structures
from autocat.adsorption import place_adsorbate
from autocat.saa import generate_saa_structures
from autocat.utils import flatten_structures_dict


def test_sequential_learner_from_json():
    # Tests generation of an SequentialLearner from a json
    sub1 = generate_surface_structures(["Au"], facets={"Au": ["110"]})["Au"]["fcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("C"))
    sub2 = generate_surface_structures(["Li"], facets={"Li": ["100"]})["Li"]["bcc100"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("Mg"))
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, Atoms("N"))
    structs = [sub1, sub2, sub3]
    labels = np.array([0.1, np.nan, 0.3])
    acds = DesignSpace(structs, labels)
    featurizer = Featurizer(SOAP, kwargs={"rcut": 5.0, "lmax": 6, "nmax": 6})
    regressor = GaussianProcessRegressor()
    predictor = Predictor(regressor=regressor, featurizer=featurizer)
    candidate_selector = CandidateSelector(
        acquisition_function="Random", num_candidates_to_pick=3
    )
    acsl = SequentialLearner(
        acds, predictor=predictor, candidate_selector=candidate_selector,
    )
    acsl.iterate()
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsl.write_json_to_disk(_tmp_dir, "testing_acsl.json")
        json_path = os.path.join(_tmp_dir, "testing_acsl.json")
        written_acsl = SequentialLearner.from_json(json_path)
        assert np.array_equal(
            written_acsl.design_space.design_space_labels,
            acds.design_space_labels,
            equal_nan=True,
        )
        assert (
            written_acsl.design_space.design_space_structures
            == acds.design_space_structures
        )
        assert written_acsl.predictor.featurizer == acsl.predictor.featurizer
        assert isinstance(written_acsl.predictor.regressor, GaussianProcessRegressor)
        assert written_acsl.candidate_selector == acsl.candidate_selector
        assert written_acsl.iteration_count == 1
        assert np.array_equal(written_acsl.train_idx, acsl.train_idx)
        assert written_acsl.train_idx[0] in [True, False]
        assert np.array_equal(written_acsl.train_idx_history, acsl.train_idx_history)
        assert written_acsl.train_idx_history[0][0] in [True, False]
        assert np.array_equal(written_acsl.predictions, acsl.predictions)
        assert np.array_equal(
            written_acsl.predictions_history, acsl.predictions_history
        )
        assert np.array_equal(written_acsl.uncertainties, acsl.uncertainties)
        assert np.array_equal(
            written_acsl.uncertainties_history, acsl.uncertainties_history
        )
        assert np.array_equal(written_acsl.candidate_indices, acsl.candidate_indices)
        assert np.array_equal(
            written_acsl.candidate_index_history, acsl.candidate_index_history
        )
        assert np.array_equal(written_acsl.acquisition_scores, acsl.acquisition_scores)


def test_sequential_learner_from_jsonified_dict():
    # Tests generating a SequentialLearner from a json dict
    sub1 = generate_surface_structures(["Au"], facets={"Au": ["110"]})["Au"]["fcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("C"))
    sub2 = generate_surface_structures(["Li"], facets={"Li": ["100"]})["Li"]["bcc100"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("Mg"))
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, Atoms("N"))
    structs = [sub1, sub2, sub3]
    encoded_structs = [atoms_encoder(struct) for struct in structs]
    labels = np.array([0.1, np.nan, 0.3])

    ds_dict = {"structures": encoded_structs, "labels": labels}

    j_dict = {"design_space": ds_dict}
    sl = SequentialLearner.from_jsonified_dict(j_dict)
    assert sl.design_space.design_space_structures == structs
    assert np.array_equal(sl.design_space.design_space_labels, labels, equal_nan=True)
    assert sl.iteration_count == 0
    assert sl.predictions is None
    assert sl.uncertainties_history is None

    # test providing sl_kwargs
    j_dict = {
        "design_space": ds_dict,
        "sl_kwargs": {"candidate_index_history": [[1]], "candidate_indices": [1]},
    }
    sl = SequentialLearner.from_jsonified_dict(j_dict)
    assert sl.candidate_indices[0] == 1
    assert sl.candidate_index_history[0][0] == 1

    # test passing through predictor and candidate selector
    feat_dict = {
        "featurizer_class": {
            "module_string": "matminer.featurizers.composition.composite",
            "class_string": "ElementProperty",
        },
        "preset": "matminer",
    }
    pred_dict = {
        "regressor": {
            "name_string": "RandomForestRegressor",
            "module_string": "sklearn.ensemble._forest",
            "kwargs": {"n_estimators": 50},
        },
        "featurizer": feat_dict,
    }
    cs_dict = {"acquisition_function": "MU"}
    j_dict = {
        "design_space": ds_dict,
        "predictor": pred_dict,
        "candidate_selector": cs_dict,
    }
    sl = SequentialLearner.from_jsonified_dict(j_dict)
    assert isinstance(sl.predictor.featurizer.featurization_object, ElementProperty)
    assert sl.predictor.regressor.n_estimators == 50
    assert sl.candidate_selector.acquisition_function == "MU"

    with pytest.raises(SequentialLearnerError):
        # catches not providing DesignSpace
        j_dict = {}
        sl = SequentialLearner.from_jsonified_dict(j_dict)


def test_sequential_learner_write_json():
    # Tests writing a SequentialLearner to disk as a json
    sub1 = generate_surface_structures(["Ag"], facets={"Ag": ["110"]})["Ag"]["fcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("B"))
    sub2 = generate_surface_structures(["Li"], facets={"Li": ["100"]})["Li"]["bcc100"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("Al"))
    sub3 = generate_surface_structures(["Ti"], facets={"Ti": ["0001"]})["Ti"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, Atoms("H"))
    structs = [sub1, sub2, sub3]
    labels = np.array([0.1, 0.2, np.nan])
    featurizer = Featurizer(featurizer_class=ElementProperty, preset="magpie")
    regressor = GaussianProcessRegressor()
    predictor = Predictor(regressor=regressor, featurizer=featurizer)
    candidate_selector = CandidateSelector(
        acquisition_function="MU", num_candidates_to_pick=2
    )
    acds = DesignSpace(structs, labels)
    acsl = SequentialLearner(
        acds, predictor=predictor, candidate_selector=candidate_selector,
    )
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsl.write_json_to_disk(_tmp_dir, "testing_acsl.json")
        with open(os.path.join(_tmp_dir, "testing_acsl.json"), "r") as f:
            sl = json.load(f)
        written_structs = [
            ase_decoder(sl["design_space"]["structures"][i]) for i in range(3)
        ]
        assert structs == written_structs
        assert np.array_equal(labels, sl["design_space"]["labels"], equal_nan=True)
        assert sl["predictor"]["featurizer"]["featurizer_class"] == {
            "module_string": "matminer.featurizers.composition.composite",
            "class_string": "ElementProperty",
        }
        assert sl["candidate_selector"] == {
            "acquisition_function": "MU",
            "num_candidates_to_pick": 2,
            "hhi_type": "production",
            "include_hhi": False,
            "include_segregation_energies": False,
            "target_window": None,
            "segregation_energy_data_source": "raban1999",
            "beta": 0.1,
            "epsilon": 0.9,
        }
        assert sl["sl_kwargs"] == {
            "iteration_count": 0,
            "train_idx": None,
            "train_idx_history": None,
            "predictions": None,
            "predictions_history": None,
            "uncertainties": None,
            "uncertainties_history": None,
            "candidate_indices": None,
            "candidate_index_history": None,
            "acquisition_scores": None,
        }

    # test after iteration
    acsl.iterate()
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsl.write_json_to_disk(_tmp_dir, "testing_acsl.json")
        with open(os.path.join(_tmp_dir, "testing_acsl.json"), "r") as f:
            sl = json.load(f)
        written_structs = [
            ase_decoder(sl["design_space"]["structures"][i]) for i in range(3)
        ]
        assert structs == written_structs
        assert np.array_equal(labels, sl["design_space"]["labels"], equal_nan=True)
        # check predictor kwargs kept
        assert sl["predictor"]["featurizer"]["featurizer_class"] == {
            "module_string": "matminer.featurizers.composition.composite",
            "class_string": "ElementProperty",
        }
        assert sl["candidate_selector"] == {
            "acquisition_function": "MU",
            "num_candidates_to_pick": 2,
            "hhi_type": "production",
            "include_hhi": False,
            "include_segregation_energies": False,
            "target_window": None,
            "segregation_energy_data_source": "raban1999",
            "beta": 0.1,
            "epsilon": 0.9,
        }
        assert sl["sl_kwargs"].get("iteration_count") == 1
        assert sl["sl_kwargs"].get("train_idx") == acsl.train_idx.tolist()
        assert sl["sl_kwargs"].get("train_idx_history") == [
            ti.tolist() for ti in acsl.train_idx_history
        ]
        assert isinstance(sl["sl_kwargs"].get("train_idx_history")[0][0], bool)
        assert sl["sl_kwargs"].get("predictions") == acsl.predictions.tolist()
        assert sl["sl_kwargs"].get("predictions_history") == [
            p.tolist() for p in acsl.predictions_history
        ]
        assert sl["sl_kwargs"].get("uncertainties") == acsl.uncertainties.tolist()
        assert sl["sl_kwargs"].get("uncertainties_history") == [
            u.tolist() for u in acsl.uncertainties_history
        ]
        assert (
            sl["sl_kwargs"].get("candidate_indices") == acsl.candidate_indices.tolist()
        )
        assert sl["sl_kwargs"].get("candidate_index_history") == [
            c.tolist() for c in acsl.candidate_index_history
        ]
        assert (
            sl["sl_kwargs"].get("acquisition_scores")
            == acsl.acquisition_scores.tolist()
        )
        assert sl["sl_kwargs"].get("acquisition_scores") is not None


def test_sequential_learner_to_jsonified_dict():
    # Tests writing a SequentialLearner to disk as a json
    sub1 = generate_surface_structures(["Ag"], facets={"Ag": ["110"]})["Ag"]["fcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("B"))
    sub2 = generate_surface_structures(["Li"], facets={"Li": ["100"]})["Li"]["bcc100"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("Al"))
    sub3 = generate_surface_structures(["Ti"], facets={"Ti": ["0001"]})["Ti"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, Atoms("H"))
    structs = [sub1, sub2, sub3]
    labels = np.array([0.1, 0.2, np.nan])
    featurizer = Featurizer(featurizer_class=ElementProperty, preset="magpie")
    regressor = GaussianProcessRegressor()
    predictor = Predictor(regressor=regressor, featurizer=featurizer)
    candidate_selector = CandidateSelector(
        acquisition_function="MU", num_candidates_to_pick=2
    )
    acds = DesignSpace(structs, labels)
    acsl = SequentialLearner(
        acds, predictor=predictor, candidate_selector=candidate_selector,
    )
    jsonified_dict = acsl.to_jsonified_dict()
    json_structs = [
        ase_decoder(jsonified_dict["design_space"]["structures"][i]) for i in range(3)
    ]
    assert structs == json_structs
    assert np.array_equal(
        labels, jsonified_dict["design_space"]["labels"], equal_nan=True
    )
    assert jsonified_dict["predictor"]["featurizer"]["featurizer_class"] == {
        "module_string": "matminer.featurizers.composition.composite",
        "class_string": "ElementProperty",
    }
    assert jsonified_dict["candidate_selector"] == {
        "acquisition_function": "MU",
        "num_candidates_to_pick": 2,
        "hhi_type": "production",
        "include_hhi": False,
        "include_segregation_energies": False,
        "target_window": None,
        "segregation_energy_data_source": "raban1999",
        "beta": 0.1,
        "epsilon": 0.9,
    }
    assert jsonified_dict["sl_kwargs"] == {
        "iteration_count": 0,
        "train_idx": None,
        "train_idx_history": None,
        "predictions": None,
        "predictions_history": None,
        "uncertainties": None,
        "uncertainties_history": None,
        "candidate_indices": None,
        "candidate_index_history": None,
        "acquisition_scores": None,
    }

    # test after iteration
    acsl.iterate()
    jsonified_dict = acsl.to_jsonified_dict()
    json_structs = [
        ase_decoder(jsonified_dict["design_space"]["structures"][i]) for i in range(3)
    ]
    assert structs == json_structs
    assert np.array_equal(
        labels, jsonified_dict["design_space"]["labels"], equal_nan=True
    )
    assert jsonified_dict["predictor"]["featurizer"]["featurizer_class"] == {
        "module_string": "matminer.featurizers.composition.composite",
        "class_string": "ElementProperty",
    }
    assert jsonified_dict["candidate_selector"] == {
        "acquisition_function": "MU",
        "num_candidates_to_pick": 2,
        "hhi_type": "production",
        "include_hhi": False,
        "include_segregation_energies": False,
        "target_window": None,
        "segregation_energy_data_source": "raban1999",
        "beta": 0.1,
        "epsilon": 0.9,
    }
    assert jsonified_dict["sl_kwargs"].get("iteration_count") == 1
    assert jsonified_dict["sl_kwargs"].get("train_idx") == acsl.train_idx.tolist()
    assert jsonified_dict["sl_kwargs"].get("train_idx_history") == [
        ti.tolist() for ti in acsl.train_idx_history
    ]
    assert isinstance(jsonified_dict["sl_kwargs"].get("train_idx_history")[0][0], bool)
    assert jsonified_dict["sl_kwargs"].get("predictions") == acsl.predictions.tolist()
    assert jsonified_dict["sl_kwargs"].get("predictions_history") == [
        p.tolist() for p in acsl.predictions_history
    ]
    assert (
        jsonified_dict["sl_kwargs"].get("uncertainties") == acsl.uncertainties.tolist()
    )
    assert jsonified_dict["sl_kwargs"].get("uncertainties_history") == [
        u.tolist() for u in acsl.uncertainties_history
    ]
    assert (
        jsonified_dict["sl_kwargs"].get("candidate_indices")
        == acsl.candidate_indices.tolist()
    )
    assert jsonified_dict["sl_kwargs"].get("candidate_index_history") == [
        c.tolist() for c in acsl.candidate_index_history
    ]
    assert (
        jsonified_dict["sl_kwargs"].get("acquisition_scores")
        == acsl.acquisition_scores.tolist()
    )
    assert jsonified_dict["sl_kwargs"].get("acquisition_scores") is not None

    # test when no uncertainty
    regressor = RandomForestRegressor()
    predictor = Predictor(regressor=regressor, featurizer=featurizer)
    candidate_selector = CandidateSelector(acquisition_function="Random")
    acsl = SequentialLearner(
        acds, predictor=predictor, candidate_selector=candidate_selector,
    )
    jsonified_dict = acsl.to_jsonified_dict()
    assert jsonified_dict["sl_kwargs"].get("uncertainties_history") == None
    assert jsonified_dict["sl_kwargs"].get("uncertainties") == None
    acsl.iterate()
    assert jsonified_dict["sl_kwargs"].get("uncertainties_history") == None
    assert jsonified_dict["sl_kwargs"].get("uncertainties") == None


def test_sequential_learner_iterate():
    # Tests iterate method
    sub1 = generate_surface_structures(["Ca"], facets={"Ca": ["111"]})["Ca"]["fcc111"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("Na"))
    sub2 = generate_surface_structures(["Nb"], facets={"Nb": ["110"]})["Nb"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("K"))
    sub3 = generate_surface_structures(["Ta"], facets={"Ta": ["110"]})["Ta"]["bcc110"][
        "structure"
    ]
    sub3 = place_adsorbate(sub3, Atoms("H"))
    sub4 = generate_surface_structures(["Sr"], facets={"Sr": ["110"]})["Sr"]["fcc110"][
        "structure"
    ]
    sub4 = place_adsorbate(sub4, Atoms("Fe"))
    structs = [sub1, sub2, sub3, sub4]
    labels = np.array([11.0, 25.0, np.nan, np.nan])
    acds = DesignSpace(structs, labels)
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    acsl = SequentialLearner(acds, predictor=predictor)

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
    assert len(acsl.train_idx_history) == 1
    assert np.count_nonzero(acsl.train_idx_history[-1]) == 2

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
    assert len(acsl.train_idx_history) == 2
    assert np.count_nonzero(acsl.train_idx_history[-1]) == 3

    acsl.design_space.update([structs[cand_ind2]], np.array([17.0]))
    acsl.iterate()

    assert acsl.iteration_count == 3
    assert acsl.candidate_structures is None
    assert acsl.candidate_indices is None
    assert acsl.candidate_index_history == [[cand_ind1], [cand_ind2]]
    assert len(acsl.uncertainties_history) == 3
    assert len(acsl.predictions_history) == 3
    assert len(acsl.train_idx_history) == 3
    assert np.count_nonzero(acsl.train_idx_history[-1]) == 4


def test_sequential_learner_setup():
    # Tests setting up an SL object
    sub1 = generate_surface_structures(["Ir"], facets={"Ir": ["100"]})["Ir"]["fcc100"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("S"))
    sub2 = generate_surface_structures(["Mo"], facets={"Mo": ["110"]})["Mo"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("H"))
    sub3 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub3 = place_adsorbate(sub3, Atoms("O"))
    sub4 = generate_surface_structures(["Re"], facets={"Re": ["0001"]})["Re"][
        "hcp0001"
    ]["structure"]
    sub4 = place_adsorbate(sub4, Atoms("N"))
    structs = [sub1, sub2, sub3, sub4]
    labels = np.array([4.0, np.nan, 6.0, np.nan])
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    acds = DesignSpace(structs, labels)
    acsl = SequentialLearner(acds, predictor=predictor)

    assert acsl.design_space.design_space_structures == acds.design_space_structures
    assert np.array_equal(
        acsl.design_space.design_space_labels, acds.design_space_labels, equal_nan=True
    )
    assert acsl.iteration_count == 0
    assert acsl.predictions == None
    assert acsl.candidate_indices == None
    assert acsl.candidate_selector == CandidateSelector()

    # test passing a candidate selector
    candidate_selector = CandidateSelector(
        acquisition_function="MU", num_candidates_to_pick=2
    )
    acsl = SequentialLearner(
        acds, predictor=predictor, candidate_selector=candidate_selector
    )
    assert acsl.candidate_selector == candidate_selector
    # ensure that a copy is made of the candidate selector
    assert acsl.candidate_selector is not candidate_selector


def test_design_space_setup():
    # test setting up an DesignSpace
    sub1 = generate_surface_structures(
        ["Pt"], supercell_dim=[2, 2, 5], facets={"Pt": ["100"]}
    )["Pt"]["fcc100"]["structure"]
    sub1 = place_adsorbate(sub1, Atoms("H"))
    sub2 = generate_surface_structures(["Na"], facets={"Na": ["110"]})["Na"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("F"))
    structs = [sub1, sub2]
    labels = np.array([3.0, 7.0])
    acds = DesignSpace(structs, labels)
    assert acds.design_space_structures == [sub1, sub2]
    assert acds.design_space_structures is not structs
    assert np.array_equal(acds.design_space_labels, labels)
    assert acds.design_space_labels is not labels
    assert len(acds) == 2
    # test different number of structures and labels
    with pytest.raises(DesignSpaceError):
        acds = DesignSpace([sub1], labels)


def test_delitem_design_space():
    # tests deleting items from the design space
    sub0 = generate_surface_structures(["Pd"], facets={"Pd": ["100"]})["Pd"]["fcc100"][
        "structure"
    ]
    sub0 = place_adsorbate(sub0, Atoms("O"))
    sub1 = generate_surface_structures(["V"], facets={"V": ["110"]})["V"]["bcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("H"))
    sub2 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("S"))
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, Atoms("P"))
    structs = [sub0, sub1, sub2]
    labels = np.array([-2.5, np.nan, 600.0])
    # test deleting by single idx
    acds = DesignSpace(structs, labels)
    del acds[1]
    assert len(acds) == 2
    assert np.array_equal(acds.design_space_labels, np.array([-2.5, 600.0]))
    assert acds.design_space_structures == [sub0, sub2]
    # test deleting using a mask
    acds = DesignSpace(structs, labels)
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
    acds = DesignSpace(structs, labels)
    del acds[[1, 3]]
    assert len(acds) == 2
    assert np.array_equal(
        acds.design_space_labels, np.array([-20, np.nan]), equal_nan=True
    )
    assert acds.design_space_structures == [sub0, sub2]
    # test deleting by providing list with a single idx
    acds = DesignSpace(structs, labels)
    del acds[[0]]
    assert len(acds) == 3
    assert np.array_equal(
        acds._design_space_labels, np.array([8, np.nan, 0.3]), equal_nan=True
    )
    assert acds.design_space_structures == [sub1, sub2, sub3]


def test_eq_design_space():
    # test comparing design spaces
    sub0 = generate_surface_structures(["Pd"], facets={"Pd": ["100"]})["Pd"]["fcc100"][
        "structure"
    ]
    sub0 = place_adsorbate(sub0, Atoms("O"))
    sub1 = generate_surface_structures(["V"], facets={"V": ["110"]})["V"]["bcc110"][
        "structure"
    ]
    sub1 = place_adsorbate(sub1, Atoms("H"))
    sub2 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("S"))
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, Atoms("P"))
    structs = [sub0, sub1, sub2]
    labels = np.array([-2.5, np.nan, 600.0])

    # test trivial case
    acds = DesignSpace(structs, labels)
    acds0 = DesignSpace(structs, labels)
    assert acds == acds0

    # test comparing when different length
    acds1 = DesignSpace(structs[:-1], labels[:-1])
    assert acds != acds1

    # test same structures, different labels
    acds2 = DesignSpace(structs, labels)
    acds2.update([structs[1]], labels=np.array([0.2]))
    assert acds != acds2

    # test diff structures, same labels
    structs[0][0].symbol = "Ni"
    acds3 = DesignSpace(structs, labels)
    assert acds != acds3


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
    acds = DesignSpace(structs, labels)

    # Test trying to update just structures
    with pytest.raises(DesignSpaceError):
        acds.design_space_structures = [sub4]

    # Test trying to update just labels
    with pytest.raises(DesignSpaceError):
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
    # Tests writing out the DesignSpace to disk
    sub1 = generate_surface_structures(["Pd"], facets={"Pd": ["111"]})["Pd"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["V"], facets={"V": ["110"]})["V"]["bcc110"][
        "structure"
    ]
    structs = [sub1, sub2]
    labels = np.array([0.3, 0.8])
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acds = DesignSpace(design_space_structures=structs, design_space_labels=labels,)
        acds.write_json_to_disk(write_location=_tmp_dir)
        # loads back written json
        with open(os.path.join(_tmp_dir, "acds.json"), "r") as f:
            ds = json.load(f)
        written_structs = [ase_decoder(ds["structures"][i]) for i in range(2)]
        assert structs == written_structs
        assert np.array_equal(labels, ds["labels"])


def test_design_space_to_jsonified_dict():
    # Tests returning the DesignSpace as a jsonified list
    sub1 = generate_surface_structures(["Pd"], facets={"Pd": ["111"]})["Pd"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["V"], facets={"V": ["110"]})["V"]["bcc110"][
        "structure"
    ]
    structs = [sub1, sub2]
    labels = np.array([0.3, 0.8])
    acds = DesignSpace(design_space_structures=structs, design_space_labels=labels,)
    jsonified_dict = acds.to_jsonified_dict()
    json_structs = [ase_decoder(jsonified_dict["structures"][i]) for i in range(2)]
    assert structs == json_structs
    assert np.array_equal(labels, jsonified_dict["labels"])


def test_design_space_from_jsonified_dict():
    # Test generating DesignSpace from a jsonified dict
    sub1 = generate_surface_structures(["Pd"], facets={"Pd": ["111"]})["Pd"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["V"], facets={"V": ["110"]})["V"]["bcc110"][
        "structure"
    ]
    structs = [sub1, sub2]
    encoded_structs = [atoms_encoder(struct) for struct in structs]
    labels = np.array([0.3, 0.8])

    with pytest.raises(DesignSpaceError):
        # catch providing only structures
        j_dict = {"structures": encoded_structs}
        ds = DesignSpace.from_jsonified_dict(j_dict)

    with pytest.raises(DesignSpaceError):
        # catch providing only labels
        j_dict = {"labels": labels}
        ds = DesignSpace.from_jsonified_dict(j_dict)

    with pytest.raises(DesignSpaceError):
        # catch structures not encoded
        j_dict = {"structures": structs, "labels": labels}
        ds = DesignSpace.from_jsonified_dict(j_dict)

    with pytest.raises(DesignSpaceError):
        # catch structures not encoded
        j_dict = {"structures": ["Pd", "V"], "labels": labels}
        ds = DesignSpace.from_jsonified_dict(j_dict)

    j_dict = {"structures": encoded_structs, "labels": labels}
    ds = DesignSpace.from_jsonified_dict(j_dict)
    assert ds.design_space_structures == structs
    assert np.array_equal(ds.design_space_labels, labels)


def test_get_design_space_from_json():
    # Tests generating DesignSpace from a json
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
        acds = DesignSpace(design_space_structures=structs, design_space_labels=labels,)
        acds.write_json_to_disk("testing.json", write_location=_tmp_dir)

        tmp_json_dir = os.path.join(_tmp_dir, "testing.json")
        acds_from_json = DesignSpace.from_json(tmp_json_dir)
        assert acds_from_json.design_space_structures == structs
        assert np.array_equal(
            acds_from_json.design_space_labels, labels, equal_nan=True
        )


def test_candidate_selector_setup():
    # Test setting up the candidate selector
    with pytest.raises(CandidateSelectorError):
        cs = CandidateSelector(
            acquisition_function="FAKE_AQ",
            num_candidates_to_pick=1,
            include_hhi=False,
            include_segregation_energies=False,
        )
    cs = CandidateSelector(
        acquisition_function="MLI",
        num_candidates_to_pick=1,
        include_hhi=False,
        include_segregation_energies=False,
        target_window=(10, -3),
    )
    assert np.array_equal(cs.target_window, (-3, 10))
    with pytest.raises(CandidateSelectorError):
        cs.target_window = (-np.inf, np.inf)
    with pytest.raises(CandidateSelectorError):
        cs = CandidateSelector(
            acquisition_function="MU",
            num_candidates_to_pick=1,
            include_hhi=True,
            include_segregation_energies=False,
            hhi_type="FAKE_HHI_TYPE",
        )
    cs = CandidateSelector(
        acquisition_function="MU",
        num_candidates_to_pick=1,
        include_hhi=False,
        include_segregation_energies=True,
        segregation_energy_data_source="raban1999",
    )
    assert cs.segregation_energy_data_source == "raban1999"
    with pytest.raises(CandidateSelectorError):
        cs = CandidateSelector(
            acquisition_function="MU",
            num_candidates_to_pick=1,
            include_hhi=False,
            include_segregation_energies=True,
            segregation_energy_data_source="FAKE_SOURCE",
        )
    cs = CandidateSelector(acquisition_function="UCB")
    assert np.isclose(cs.beta, 0.1)
    cs = CandidateSelector(acquisition_function="UCB", beta=0.4)
    assert np.isclose(cs.beta, 0.4)
    cs = CandidateSelector(acquisition_function="LCBAdaptive")
    assert np.isclose(cs.beta, 3)
    assert np.isclose(cs.epsilon, 0.9)


def test_candidate_selector_from_jsonified_dict():
    # Tests generating CandidateSelector from a json dict

    # test defaults
    j_dict = {}
    cs = CandidateSelector.from_jsonified_dict(j_dict)
    assert cs.acquisition_function == "Random"
    assert cs.num_candidates_to_pick == 1

    # test specifying parameters
    j_dict = {
        "acquisition_function": "MLI",
        "target_window": (-np.inf, 50.0),
        "include_hhi": True,
        "hhi_type": "reserves",
        "include_segregation_energies": True,
        "segregation_energy_data_source": "rao2020",
        "beta": 0.75,
        "epsilon": 0.4,
    }
    cs = CandidateSelector.from_jsonified_dict(j_dict)
    assert cs.acquisition_function == "MLI"
    assert np.array_equal(cs.target_window, [-np.inf, 50.0])
    assert cs.include_hhi
    assert cs.include_segregation_energies
    assert cs.hhi_type == "reserves"
    assert cs.segregation_energy_data_source == "rao2020"
    assert np.isclose(cs.beta, 0.75)
    assert np.isclose(cs.epsilon, 0.4)


def test_candidate_selector_to_jsonified_dict():
    # Tests converting CandidateSelector to json dict
    cs = CandidateSelector(
        acquisition_function="MLI",
        target_window=(-10.0, np.inf),
        include_hhi=True,
        hhi_type="reserves",
        include_segregation_energies=True,
        segregation_energy_data_source="rao2020",
        beta=0.05,
        epsilon=4,
    )
    conv_j_dict = cs.to_jsonified_dict()
    assert conv_j_dict.get("acquisition_function") == "MLI"
    assert conv_j_dict.get("include_hhi")
    assert conv_j_dict.get("include_segregation_energies")
    assert conv_j_dict.get("hhi_type") == "reserves"
    assert conv_j_dict.get("segregation_energy_data_source") == "rao2020"
    assert np.array_equal(conv_j_dict.get("target_window"), [-10.0, np.inf])
    assert np.isclose(conv_j_dict.get("beta"), 0.05)
    assert np.isclose(conv_j_dict.get("epsilon"), 4)


def test_write_candidate_selector_as_json():
    # Test writing out CandidateSelector to disk as a json
    with tempfile.TemporaryDirectory() as _tmp_dir:
        cs = CandidateSelector(
            acquisition_function="MU",
            include_hhi=True,
            hhi_type="reserves",
            target_window=[-np.inf, 0.0],
            beta=4,
            epsilon=0.3,
        )
        cs.write_json_to_disk(write_location=_tmp_dir, json_name="cs.json")
        # loads back written json
        with open(os.path.join(_tmp_dir, "cs.json"), "r") as f:
            written_cs = json.load(f)
        assert written_cs.get("acquisition_function") == "MU"
        assert np.array_equal(written_cs.get("target_window"), [-np.inf, 0.0])
        assert written_cs.get("include_hhi")
        assert written_cs.get("hhi_type") == "reserves"
        assert np.isclose(written_cs.get("beta"), 4)
        assert np.isclose(written_cs.get("epsilon"), 0.3)


def test_get_candidate_selector_from_json():
    # Tests generating a CandidateSelector from a json
    with tempfile.TemporaryDirectory() as _tmp_dir:
        cs = CandidateSelector(
            acquisition_function="MLI",
            include_segregation_energies=True,
            target_window=(-3.0, np.inf),
            segregation_energy_data_source="rao2020",
        )
        cs.write_json_to_disk(json_name="testing.json", write_location=_tmp_dir)

        tmp_json_dir = os.path.join(_tmp_dir, "testing.json")
        cs_from_json = CandidateSelector.from_json(tmp_json_dir)
        assert cs_from_json.acquisition_function == "MLI"
        assert cs_from_json.include_segregation_energies
        assert not cs_from_json.include_hhi
        assert cs_from_json.segregation_energy_data_source == "rao2020"
        assert np.array_equal(cs_from_json.target_window, [-3.0, np.inf])


def test_candidate_selector_eq():
    # Tests CandidateSelector equality
    cs = CandidateSelector(
        acquisition_function="MLI",
        num_candidates_to_pick=1,
        include_hhi=True,
        hhi_type="reserves",
        include_segregation_energies=True,
        target_window=(-100, -30),
        segregation_energy_data_source="raban1999",
    )

    cs2 = CandidateSelector(
        acquisition_function="MLI",
        num_candidates_to_pick=1,
        include_hhi=True,
        hhi_type="reserves",
        include_segregation_energies=True,
        target_window=(-100, -30),
        segregation_energy_data_source="raban1999",
    )

    assert cs == cs2

    cs2.target_window = (0, 3)
    assert not cs == cs2
    cs2.target_window = (-100, -30)

    cs2.include_hhi = False
    assert not cs == cs2
    cs2.include_hhi = True

    cs2.hhi_type = "production"
    assert not cs == cs2
    cs2.hhi_type = "reserves"

    cs2.include_segregation_energies = False
    assert not cs == cs2
    cs2.include_segregation_energies = True

    cs2.segregation_energy_data_source = "rao2020"
    assert not cs == cs2
    cs2.segregation_energy_data_source = "raban1999"

    cs2.beta = 0.8
    assert not cs == cs2
    cs2.beta = 0.1

    cs2.epsilon = 0.1
    assert not cs == cs2


def test_candidate_selector_copy():
    # Tests making a copy of a CandidateSelector
    cs = CandidateSelector(
        acquisition_function="MLI",
        num_candidates_to_pick=5,
        include_hhi=True,
        hhi_type="reserves",
        include_segregation_energies=True,
        target_window=(1, 40),
    )

    cs2 = cs.copy()
    assert cs == cs2
    assert not cs is cs2
    assert not cs.target_window is cs2.target_window


def test_candidate_selector_choose_candidate():
    # Test choosing candidates with CandidateSelector
    # (without segregation energy or hhi weighting)
    sub1 = generate_surface_structures(
        ["Pt"], supercell_dim=[2, 2, 5], facets={"Pt": ["100"]}
    )["Pt"]["fcc100"]["structure"]
    sub1 = place_adsorbate(sub1, Atoms("H"))
    sub2 = generate_surface_structures(["Na"], facets={"Na": ["110"]})["Na"]["bcc110"][
        "structure"
    ]
    sub2 = place_adsorbate(sub2, Atoms("H"))
    sub3 = generate_surface_structures(["Ru"], facets={"Ru": ["0001"]})["Ru"][
        "hcp0001"
    ]["structure"]
    sub3 = place_adsorbate(sub3, Atoms("H"))
    sub4 = generate_surface_structures(["Li"], facets={"Li": ["110"]})["Li"]["bcc110"][
        "structure"
    ]
    sub4 = place_adsorbate(sub4, Atoms("H"))
    structs = [sub1, sub2, sub3, sub4]
    labels = np.array([3.0, np.nan, np.nan, np.nan])
    ds = DesignSpace(structs, labels)
    unc = np.array([0.1, 0.2, 0.5, 0.3])
    cs = CandidateSelector(
        acquisition_function="MU",
        num_candidates_to_pick=1,
        include_hhi=False,
        include_segregation_energies=False,
    )
    # automatically chooses among systems without labels
    parent_idx, max_score, aq_scores = cs.choose_candidate(
        design_space=ds, uncertainties=unc
    )
    assert parent_idx[0] == 2
    # ensure picking system with highest aq score
    for i in range(1, 4):
        if i != parent_idx:
            assert aq_scores[i] < max_score

    # multiple candidates to pick
    cs.num_candidates_to_pick = 2
    parent_idx, max_scores, aq_scores = cs.choose_candidate(
        design_space=ds, uncertainties=unc
    )
    assert np.array_equal(parent_idx, [3, 2])
    assert len(max_scores) == 2
    # ensure picking systems with highest aq scores
    min_max_score = min(max_scores)
    for i in range(1, 4):
        if i not in parent_idx:
            assert aq_scores[i] < min_max_score

    cs.num_candidates_to_pick = 1

    # restrict indices to choose from
    allowed_idx = np.array([0, 1, 0, 1], dtype=bool)
    parent_idx, _, _ = cs.choose_candidate(
        design_space=ds, uncertainties=unc, allowed_idx=allowed_idx
    )
    assert parent_idx[0] == 3

    # need uncertainty for MU
    with pytest.raises(CandidateSelectorError):
        parent_idx, _, _ = cs.choose_candidate(design_space=ds)

    # fully explored ds
    labels = np.array([3.0, 4.0, 5.0, 10.0])
    ds2 = DesignSpace(structs, labels)
    parent_idx, _, _ = cs.choose_candidate(design_space=ds2, uncertainties=unc)
    assert parent_idx[0] == 2

    cs.acquisition_function = "MLI"
    cs.target_window = (-np.inf, 0.15)
    pred = np.array([3.0, 0.3, 6.0, 9.0])
    # need both uncertainty and predictions for MLI
    with pytest.raises(CandidateSelectorError):
        parent_idx, _, _ = cs.choose_candidate(design_space=ds, uncertainties=unc)
    parent_idx, _, _ = cs.choose_candidate(
        design_space=ds, uncertainties=unc, predictions=pred
    )
    assert parent_idx[0] == 1

    # test UCB
    cs.acquisition_function = "UCB"
    cs.beta = 0.2
    # need both uncertainty and predictions for UCB
    with pytest.raises(CandidateSelectorError):
        parent_idx, _, _ = cs.choose_candidate(design_space=ds, uncertainties=unc)
    pred2 = np.array([3.0, 0.3, 8.9, 9.0])
    unc2 = np.array([0.1, 0.2, 1.0, 0.3])
    parent_idx, _, _ = cs.choose_candidate(
        design_space=ds, predictions=pred2, uncertainties=unc2
    )
    assert parent_idx[0] == 2

    # test LCB
    cs.acquisition_function = "LCB"
    cs.beta = 0.8
    # need both uncertainty and predictions for LCB
    with pytest.raises(CandidateSelectorError):
        parent_idx, _, _ = cs.choose_candidate(design_space=ds, uncertainties=unc)
    pred3 = np.array([3.0, 0.3, 8.9, 9.0])
    unc3 = np.array([0.1, 0.2, 1.0, 0.3])
    parent_idx, _, _ = cs.choose_candidate(
        design_space=ds, predictions=pred3, uncertainties=unc3
    )
    assert parent_idx[0] == 3

    # test LCBAdaptive
    cs.acquisition_function = "LCBAdaptive"
    cs.beta = 9
    cs.epsilon = 0.9
    # need both uncertainty and predictions for LCBAdaptive
    with pytest.raises(CandidateSelectorError):
        parent_idx, _, _ = cs.choose_candidate(design_space=ds, uncertainties=unc)
    # needs iteration count for LCBAdaptive
    with pytest.raises(CandidateSelectorError):
        parent_idx, _, _ = cs.choose_candidate(
            design_space=ds, uncertainties=unc, predictions=pred
        )
    pred4 = np.array([3.0, 0.3, 20.0, 10.0])
    unc4 = np.array([0.1, 0.2, 5.0, 0.03])
    parent_idx, _, _ = cs.choose_candidate(
        design_space=ds,
        predictions=pred4,
        uncertainties=unc4,
        number_of_labelled_data_pts=1,
    )
    assert parent_idx[0] == 3
    parent_idx, _, _ = cs.choose_candidate(
        design_space=ds,
        predictions=pred4,
        uncertainties=unc4,
        number_of_labelled_data_pts=30,
    )
    assert parent_idx[0] == 2


def test_candidate_selector_choose_candidate_hhi_weighting():
    # Tests that the HHI weighting is properly applied
    unc = np.array([0.1, 0.1])
    pred = np.array([4.0, 4.0])
    labels = np.array([np.nan, np.nan])
    # Tests using production HHI values and MU
    y_struct = generate_surface_structures(["Y"], facets={"Y": ["0001"]})["Y"][
        "hcp0001"
    ]["structure"]
    ni_struct = generate_surface_structures(["Ni"], facets={"Ni": ["111"]})["Ni"][
        "fcc111"
    ]["structure"]
    ds = DesignSpace([y_struct, ni_struct], labels)
    cs = CandidateSelector(
        acquisition_function="MU",
        num_candidates_to_pick=1,
        include_hhi=True,
        include_segregation_energies=False,
    )
    parent_idx, _, aq_scores = cs.choose_candidate(design_space=ds, uncertainties=unc)
    assert parent_idx[0] == 1
    assert aq_scores[0] < aq_scores[1]

    # Tests using reserves HHI values and MLI
    nb_struct = generate_surface_structures(["Nb"], facets={"Nb": ["111"]})["Nb"][
        "bcc111"
    ]["structure"]
    na_struct = generate_surface_structures(["Na"], facets={"Na": ["110"]})["Na"][
        "bcc110"
    ]["structure"]
    ds = DesignSpace([na_struct, nb_struct], labels)
    cs.acquisition_function = "MLI"
    cs.target_window = (3, 5)
    cs.hhi_type = "reserves"
    parent_idx, _, aq_scores = cs.choose_candidate(
        design_space=ds, uncertainties=unc, predictions=pred
    )
    assert parent_idx[0] == 0
    assert aq_scores[0] > aq_scores[1]


def test_candidate_selector_choose_candidate_segregation_energy_weighting():
    # Tests that the segregation energy weighting is properly applied
    unc = np.array([0.3, 0.3])
    pred = np.array([2.0, 2.0])
    labels = np.array([np.nan, np.nan])
    structs = flatten_structures_dict(
        generate_saa_structures(["Cr"], ["Rh"], facets={"Cr": ["110"]})
    )
    structs.extend(
        flatten_structures_dict(
            generate_saa_structures(["Co"], ["Re"], facets={"Co": ["0001"]})
        )
    )
    ds = DesignSpace(structs, labels)
    cs = CandidateSelector(
        acquisition_function="MLI",
        target_window=(0, 4),
        include_hhi=False,
        include_segregation_energies=True,
    )
    parent_idx, _, aq_scores = cs.choose_candidate(
        design_space=ds, uncertainties=unc, predictions=pred
    )
    assert parent_idx[0] == 0
    assert aq_scores[0] > aq_scores[1]


def test_simulated_sequential_histories():
    # Test output sl has appropriate histories
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, Atoms("O"))
    base_struct2 = place_adsorbate(sub2, Atoms("N"))
    base_struct3 = place_adsorbate(sub2, Atoms("H"))
    ds_structs = [
        base_struct1,
        base_struct2,
        base_struct3,
        sub1,
        sub2,
    ]
    ds_labels = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    acds = DesignSpace(ds_structs, ds_labels)
    candidate_selector = CandidateSelector(
        acquisition_function="MLI", num_candidates_to_pick=2, target_window=(0.9, 2.1)
    )
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    sl = simulated_sequential_learning(
        full_design_space=acds,
        init_training_size=1,
        number_of_sl_loops=2,
        candidate_selector=candidate_selector,
        predictor=predictor,
    )

    # Test number of sl loops
    assert sl.iteration_count == 3

    # Test initial training size
    assert sl.train_idx_history[0].sum() == 1

    # Test keeping track of pred and unc history
    assert len(sl.uncertainties_history) == 3
    assert len(sl.uncertainties_history[0]) == len(acds)
    assert len(sl.predictions_history) == 3
    assert len(sl.predictions_history[-1]) == len(acds)
    assert len(sl.candidate_index_history) == 2


def test_simulated_sequential_batch_added():
    # Tests adding N candidates on each loop
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, Atoms("O"))
    base_struct2 = place_adsorbate(sub2, Atoms("N"))
    candidate_selector = CandidateSelector(
        acquisition_function="Random", num_candidates_to_pick=2
    )
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    num_loops = 2
    ds_structs = [base_struct1, base_struct2, sub1, sub2]
    ds_labels = np.array([5.0, 6.0, 7.0, 8.0])
    acds = DesignSpace(ds_structs, ds_labels)
    sl = simulated_sequential_learning(
        full_design_space=acds,
        predictor=predictor,
        candidate_selector=candidate_selector,
        number_of_sl_loops=num_loops,
        init_training_size=1,
    )
    # should add 2 candidates on first loop
    assert len(sl.candidate_index_history[0]) == 2
    # since only 1 left, should add it on the next
    assert len(sl.candidate_index_history[1]) == 1


def test_simulated_sequential_num_loops():
    # Tests the number of loops
    sub1 = generate_surface_structures(["Fe"], facets={"Fe": ["110"]})["Fe"]["bcc110"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, Atoms("H"))
    base_struct2 = place_adsorbate(sub2, Atoms("N"))
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    candidate_selector = CandidateSelector(
        acquisition_function="Random", num_candidates_to_pick=3
    )
    ds_structs = [base_struct1, base_struct2, sub1, sub2]
    ds_labels = np.array([5.0, 6.0, 7.0, 8.0])
    acds = DesignSpace(ds_structs, ds_labels)
    # Test default number of loops
    sl = simulated_sequential_learning(
        full_design_space=acds,
        predictor=predictor,
        candidate_selector=candidate_selector,
        init_training_size=1,
    )
    assert len(sl.predictions_history) == 2
    assert sl.iteration_count == 2

    # Test catches maximum number of loops
    with pytest.raises(SequentialLearnerError):
        sl = simulated_sequential_learning(
            full_design_space=acds,
            predictor=predictor,
            candidate_selector=candidate_selector,
            init_training_size=1,
            number_of_sl_loops=3,
        )

    # Test with default num loops and default num candidates
    ds_structs = [base_struct1, base_struct2, sub2]
    ds_labels = np.array([5.0, 6.0, 7.0])
    acds = DesignSpace(ds_structs, ds_labels)
    candidate_selector.num_candidates_to_pick = 1

    sl = simulated_sequential_learning(
        full_design_space=acds,
        predictor=predictor,
        candidate_selector=candidate_selector,
        init_training_size=1,
    )
    assert len(sl.uncertainties_history) == 3
    assert sl.iteration_count == 3


def test_simulated_sequential_write_to_disk():
    # Test writing out sl dict
    with tempfile.TemporaryDirectory() as _tmp_dir:
        sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"][
            "fcc111"
        ]["structure"]
        sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"][
            "fcc100"
        ]["structure"]
        base_struct1 = place_adsorbate(sub1, Atoms("O"))
        base_struct2 = place_adsorbate(sub2, Atoms("S"))
        base_struct3 = place_adsorbate(sub2, Atoms("N"))
        featurizer = Featurizer(featurizer_class=SineMatrix)
        regressor = RandomForestRegressor(n_estimators=125)
        predictor = Predictor(featurizer=featurizer, regressor=regressor)
        candidate_selector = CandidateSelector(
            acquisition_function="Random", num_candidates_to_pick=2
        )
        ds_structs = [base_struct1, base_struct2, base_struct3]
        ds_labels = np.array([0, 1, 2])
        acds = DesignSpace(ds_structs, ds_labels)
        sl = simulated_sequential_learning(
            full_design_space=acds,
            init_training_size=2,
            number_of_sl_loops=1,
            predictor=predictor,
            candidate_selector=candidate_selector,
            write_to_disk=True,
            write_location=_tmp_dir,
        )
        # check data written as json
        json_path = os.path.join(_tmp_dir, "sequential_learner.json")
        sl_written = SequentialLearner.from_json(json_path)
        assert sl.iteration_count == sl_written.iteration_count
        assert np.array_equal(sl.predictions_history, sl_written.predictions_history)
        assert np.array_equal(
            sl.uncertainties_history, sl_written.uncertainties_history
        )
        assert np.array_equal(
            sl.candidate_index_history, sl_written.candidate_index_history
        )
        assert np.array_equal(sl.candidate_indices, sl_written.candidate_indices)
        assert np.array_equal(sl.predictions, sl_written.predictions)
        assert np.array_equal(sl.uncertainties, sl_written.uncertainties)
        assert sl_written.predictor.featurizer == sl.predictor.featurizer
        assert isinstance(sl_written.predictor.regressor, RandomForestRegressor)
        assert sl_written.predictor.regressor.n_estimators == 125
        assert sl.candidate_selector == sl_written.candidate_selector
        assert (
            sl.design_space.design_space_structures
            == sl_written.design_space.design_space_structures
        )
        assert np.array_equal(
            sl.design_space.design_space_labels,
            sl_written.design_space.design_space_labels,
        )


def test_simulated_sequential_learning_fully_explored():
    # Checks that catches if ds not fully explored
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, Atoms("OH"))
    base_struct2 = place_adsorbate(sub2, Atoms("NH"))
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    ds_structs = [base_struct1, base_struct2, sub2]
    ds_labels = np.array([0.0, np.nan, 4.0])
    acds = DesignSpace(ds_structs, ds_labels)
    candidate_selector = CandidateSelector(acquisition_function="MU")
    with pytest.raises(SequentialLearnerError):
        sl = simulated_sequential_learning(
            full_design_space=acds,
            init_training_size=1,
            number_of_sl_loops=2,
            predictor=predictor,
            candidate_selector=candidate_selector,
        )


def test_multiple_sequential_learning_serial():
    # Tests serial implementation
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, Atoms("O"))
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    ds_structs = [base_struct1, sub1]
    ds_labels = np.array([0.0, 0.0])
    acds = DesignSpace(ds_structs, ds_labels)
    candidate_selector = CandidateSelector(acquisition_function="MU")
    runs_history = multiple_simulated_sequential_learning_runs(
        full_design_space=acds,
        number_of_runs=3,
        predictor=predictor,
        candidate_selector=candidate_selector,
        number_of_sl_loops=1,
        init_training_size=1,
    )
    assert len(runs_history) == 3
    assert isinstance(runs_history[0], SequentialLearner)
    assert len(runs_history[1].predictions_history) == 2


def test_multiple_sequential_learning_parallel():
    # Tests parallel implementation
    sub1 = generate_surface_structures(["Cu"], facets={"Cu": ["111"]})["Cu"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, Atoms("Li"))
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    ds_structs = [base_struct1, sub1]
    ds_labels = np.array([0.0, 0.0])
    acds = DesignSpace(ds_structs, ds_labels)
    candidate_selector = CandidateSelector()
    runs_history = multiple_simulated_sequential_learning_runs(
        full_design_space=acds,
        number_of_runs=3,
        number_parallel_jobs=2,
        predictor=predictor,
        candidate_selector=candidate_selector,
        number_of_sl_loops=1,
        init_training_size=1,
    )
    assert len(runs_history) == 3
    assert isinstance(runs_history[2], SequentialLearner)
    assert len(runs_history[1].uncertainties_history) == 2


def test_multiple_sequential_learning_write_to_disk():
    # Tests writing run history to disk
    _tmp_dir = tempfile.TemporaryDirectory().name
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, Atoms("N"))
    featurizer = Featurizer(featurizer_class=SineMatrix)
    regressor = GaussianProcessRegressor()
    predictor = Predictor(featurizer=featurizer, regressor=regressor)
    ds_structs = [base_struct1, sub1]
    ds_labels = np.array([0.0, 0.0])
    acds = DesignSpace(ds_structs, ds_labels)
    candidate_selector = CandidateSelector(
        acquisition_function="Random", num_candidates_to_pick=2
    )
    runs_history = multiple_simulated_sequential_learning_runs(
        full_design_space=acds,
        predictor=predictor,
        candidate_selector=candidate_selector,
        number_of_runs=3,
        number_parallel_jobs=2,
        init_training_size=1,
        number_of_sl_loops=1,
        write_to_disk=True,
        write_location=_tmp_dir,
        json_name_prefix="test_multi",
    )

    # check data history in each run
    for i in range(3):
        written_run = SequentialLearner.from_json(
            os.path.join(_tmp_dir, f"test_multi_{i}.json")
        )
        written_ds = written_run.design_space
        assert written_ds.design_space_structures == ds_structs
        assert np.array_equal(written_ds.design_space_labels, ds_labels)
        assert written_run.iteration_count == runs_history[i].iteration_count
        assert np.array_equal(written_run.predictions, runs_history[i].predictions)
        assert np.array_equal(
            written_run.predictions_history, runs_history[i].predictions_history
        )
        assert np.array_equal(written_run.uncertainties, runs_history[i].uncertainties)
        assert np.array_equal(
            written_run.uncertainties_history, runs_history[i].uncertainties_history
        )
        assert np.array_equal(
            written_run.train_idx_history, runs_history[i].train_idx_history
        )
        assert np.array_equal(written_run.train_idx, runs_history[i].train_idx)
        assert np.array_equal(
            written_run.candidate_indices, runs_history[i].candidate_indices
        )
        assert np.array_equal(
            written_run.candidate_index_history, runs_history[i].candidate_index_history
        )
        assert written_run.candidate_selector == runs_history[i].candidate_selector


def test_get_overlap_score():
    # Tests default behavior
    mean = 0.0
    std = 0.1
    x1 = -0.4
    x2 = 0.8
    norm = stats.norm(loc=mean, scale=std)

    # checks that at least target min or max is provided
    with pytest.raises(SequentialLearnerError):
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
        el: 1.0 - (HHI["production"][el] - 500.0) / 9300.0 for el in HHI["production"]
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
        el: 1.0 - (HHI["reserves"][el] - 500.0) / 8600.0 for el in HHI["reserves"]
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
    # check exclude species
    hhi_exclude_scores = calculate_hhi_scores(saa_structs, "reserves", ["Ru"])
    assert np.isclose(hhi_exclude_scores[0], norm_hhi_res["Pt"])
    assert np.isclose(hhi_exclude_scores[1], norm_hhi_res["Cu"])
    assert np.isclose(hhi_exclude_scores[2], norm_hhi_res["Ni"])
    ads_struct = place_adsorbate(saa_structs[0], Atoms("Li"))
    hhi_ads_prod = calculate_hhi_scores(
        [ads_struct], "production", exclude_species=["Li"]
    )
    assert np.isclose(hhi_ads_prod[0], hhi_prod_scores[0])


def test_calculate_segregation_energy_scores():
    # Tests calculating segregation energy scores
    saa_structs = flatten_structures_dict(
        generate_saa_structures(
            ["Ag", "Ni"], ["Pt"], facets={"Ag": ["111"], "Ni": ["111"]},
        )
    )
    saa_structs.extend(
        flatten_structures_dict(
            generate_saa_structures(["Pd"], ["W"], facets={"Pd": ["111"]})
        )
    )
    # test calculating scores from RABAN1999
    se_scores = calculate_segregation_energy_scores(saa_structs)
    assert np.isclose(se_scores[-1], 0.0)
    min_seg = SEGREGATION_ENERGIES["raban1999"]["Fe_100"]["Ag"]
    max_seg = SEGREGATION_ENERGIES["raban1999"]["Pd"]["W"]
    assert np.isclose(
        se_scores[0],
        1.0
        - (SEGREGATION_ENERGIES["raban1999"]["Ag"]["Pt"] - min_seg)
        / (max_seg - min_seg),
    )
    assert np.isclose(
        se_scores[1],
        1.0
        - (SEGREGATION_ENERGIES["raban1999"]["Ni"]["Pt"] - min_seg)
        / (max_seg - min_seg),
    )

    # test getting scores from RAO2020
    se_scores = calculate_segregation_energy_scores(saa_structs, data_source="rao2020")
    assert np.isclose(se_scores[0], SEGREGATION_ENERGIES["rao2020"]["Ag"]["Pt"])
    assert np.isclose(se_scores[0], 0.8)
    assert np.isclose(se_scores[1], SEGREGATION_ENERGIES["rao2020"]["Ni"]["Pt"])
    assert np.isclose(se_scores[1], 1.0)
    assert np.isclose(se_scores[-1], SEGREGATION_ENERGIES["rao2020"]["Pd"]["W"])
    assert np.isclose(se_scores[-1], 0.0)
