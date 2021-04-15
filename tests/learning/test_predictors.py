"""Unit tests for the `autocat.learning.predictors` module"""

import os
import pytest
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor

from ase import Atoms

from autocat.adsorption import place_adsorbate
from autocat.surface import generate_surface_structures
from autocat.perturbations import generate_perturbed_dataset
from autocat.learning.featurizers import get_X
from autocat.learning.featurizers import catalyst_featurization
from autocat.learning.predictors import AutoCatStructureCorrector
from autocat.learning.predictors import AutocatStructureCorrectorError


def test_fit_model_on_perturbed_systems():
    # Test returns a fit model
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "OH")["custom"]["structure"]
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=15,)
    p_structures = p_set["collected_structures"]
    correction_matrix = p_set["correction_matrix"]
    acsc = AutoCatStructureCorrector(
        structure_featurizer="sine_matrix",
        adsorbate_featurizer="soap",
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    acsc.fit(
        p_structures, correction_matrix=correction_matrix,
    )
    assert acsc.adsorbate_featurizer == "soap"
    assert acsc.is_fit

    # check no longer fit after changing setting
    acsc.adsorbate_featurizer = "acsf"
    assert acsc.adsorbate_featurizer == "acsf"
    assert not acsc.is_fit

    # check can be fit on corrections list
    p_set["corrections_list"]
    acsc.adsorbate_featurizer = "soap"
    acsc.adsorbate_featurization_kwargs = {"rcut": 3.0, "nmax": 6, "lmax": 6}
    acsc.fit(
        p_structures, correction_matrix=correction_matrix,
    )
    assert acsc.is_fit
    with pytest.raises(AutocatStructureCorrectorError):
        acsc.fit(p_structures)


def test_predict_initial_configuration_formats():
    # Test outputs are returned as expected
    sub = generate_surface_structures(["Fe"], facets={"Fe": ["100"]})["Fe"]["bcc100"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "CO")["custom"]["structure"]
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=20,)
    p_structures = p_set["collected_structures"]
    correction_matrix = p_set["correction_matrix"]
    acsc = AutoCatStructureCorrector(
        structure_featurizer="sine_matrix",
        adsorbate_featurizer="soap",
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    acsc.fit(
        p_structures[:15], correction_matrix=correction_matrix[:15, :],
    )
    predicted_corrections, corrected_structures, uncs = acsc.predict(p_structures[15:],)
    assert isinstance(corrected_structures[0], Atoms)
    assert len(corrected_structures) == 5
    # check that even with refining, corrected structure is
    # returned to full size
    assert len(corrected_structures[2]) == len(p_structures[17])
    assert len(predicted_corrections) == 5
    # check that predicted correction matrix is applied correctly
    manual = p_structures[15].copy()
    manual_corr_mat = predicted_corrections[0]
    assert predicted_corrections[0].shape == (2, 3)
    manual.positions[-1] += manual_corr_mat[-1]
    manual.positions[-2] += manual_corr_mat[-2]
    assert np.allclose(manual.positions, corrected_structures[0].positions)
    # check dimension of uncertainty estimates
    assert len(uncs) == 5


def test_score_on_perturbed_systems():
    # Tests that the score metric yields floats
    sub = generate_surface_structures(["Fe"], facets={"Fe": ["100"]})["Fe"]["bcc100"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "CO")["custom"]["structure"]
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=20,)
    p_structures = p_set["collected_structures"]
    correction_matrix = p_set["correction_matrix"]
    corrections_list = p_set["corrections_list"]
    acsc = AutoCatStructureCorrector(
        structure_featurizer="sine_matrix",
        adsorbate_featurizer="soap",
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    acsc.fit(
        p_structures[:15], correction_matrix=correction_matrix[:15, :],
    )
    mae = acsc.score(p_structures[15:], corrections_list)
    assert isinstance(mae, float)
    rmse = acsc.score(p_structures[15:], corrections_list, metric="rmse")
    assert mae != rmse
    with pytest.raises(AutocatStructureCorrectorError):
        acsc.score(p_structures[15:], corrections_list, metric="msd")


def test_model_class_and_kwargs():
    # Tests providing regression model class and kwargs
    acsc = AutoCatStructureCorrector(KernelRidge, model_kwargs={"gamma": 0.5})
    assert isinstance(acsc.regressor, KernelRidge)
    # check that regressor created with correct kwarg
    assert acsc.regressor.gamma == 0.5
    assert acsc.model_kwargs == {"gamma": 0.5}
    acsc.model_class = GaussianProcessRegressor
    # check that kwargs are removed when class is changed
    assert acsc.model_kwargs is None
    acsc = AutoCatStructureCorrector()
    acsc.model_kwargs = {"alpha": 2.5}
    assert acsc.model_kwargs == {"alpha": 2.5}
    assert acsc.regressor.alpha == 2.5
