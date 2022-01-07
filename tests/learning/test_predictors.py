"""Unit tests for the `autocat.learning.predictors` module"""

import os
import pytest
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge

from ase import Atoms

from autocat.adsorption import place_adsorbate
from autocat.surface import generate_surface_structures
from autocat.perturbations import generate_perturbed_dataset
from autocat.learning.featurizers import get_X
from autocat.learning.featurizers import catalyst_featurization
from autocat.learning.predictors import Predictor
from autocat.learning.predictors import PredictorError


def test_fit_model_on_perturbed_systems():
    # Test returns a fit model
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "OH")["custom"]["structure"]
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=15,)
    p_structures = p_set["collected_structures"]
    correction_matrix = p_set["correction_matrix"]
    acsc = Predictor(
        structure_featurizer="sine_matrix",
        adsorbate_featurizer="soap",
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    acsc.fit(
        p_structures, y=correction_matrix,
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
        p_structures, y=correction_matrix,
    )
    assert acsc.is_fit


def test_predict_initial_configuration_formats():
    # Test outputs are returned as expected
    sub = generate_surface_structures(["Fe"], facets={"Fe": ["100"]})["Fe"]["bcc100"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "CO")["custom"]["structure"]
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=20,)
    p_structures = p_set["collected_structures"]
    correction_matrix = p_set["correction_matrix"]
    acsc = Predictor(
        structure_featurizer="sine_matrix",
        adsorbate_featurizer="soap",
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    acsc.fit(
        p_structures[:15], correction_matrix[:15, :],
    )
    predicted_corrections, uncs = acsc.predict(p_structures[15:],)
    assert len(predicted_corrections) == 5
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
    correction_matrix = p_set["correction_matrix"][:, 0]
    acsc = Predictor(
        structure_featurizer="sine_matrix",
        adsorbate_featurizer="soap",
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    acsc.fit(
        p_structures[:15], correction_matrix[:15],
    )
    mae = acsc.score(p_structures[15:], correction_matrix[15:])
    assert isinstance(mae, float)
    mse = acsc.score(p_structures[15:], correction_matrix[15:], metric="mse")
    assert isinstance(mse, float)

    # Test returning predictions
    score, pred_corr, unc = acsc.score(
        p_structures[15:], correction_matrix[15:], return_predictions=True
    )
    assert len(pred_corr) == 5
    assert len(unc) == 5
    assert mae != mse
    with pytest.raises(PredictorError):
        acsc.score(p_structures[15:], correction_matrix, metric="msd")

    # Test with single target
    acsc.fit(
        p_structures[:15], np.arange(15),
    )

    mae = acsc.score(p_structures[15:], np.arange(5))
    assert isinstance(mae, float)


def test_model_class_and_kwargs():
    # Tests providing regression model class and kwargs
    acsc = Predictor(KernelRidge, model_kwargs={"gamma": 0.5})
    assert isinstance(acsc.regressor, KernelRidge)
    # check that regressor created with correct kwarg
    assert acsc.regressor.gamma == 0.5
    assert acsc.model_kwargs == {"gamma": 0.5}
    acsc.model_class = GaussianProcessRegressor
    # check that kwargs are removed when class is changed
    assert acsc.model_kwargs is None
    acsc = Predictor()
    acsc.model_kwargs = {"alpha": 2.5}
    assert acsc.model_kwargs == {"alpha": 2.5}
    assert acsc.regressor.alpha == 2.5


def test_model_without_unc():
    # Test that predictions are still made when the model class
    # provided does not have uncertainty
    sub = generate_surface_structures(["Li"], facets={"Li": ["100"]})["Li"]["bcc100"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "S")["custom"]["structure"]
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=20,)
    p_structures = p_set["collected_structures"]
    correction_matrix = p_set["correction_matrix"]
    acsc = Predictor(
        model_class=KernelRidge,
        structure_featurizer="sine_matrix",
        adsorbate_featurizer=None,
    )
    acsc.fit(
        p_structures[:15], correction_matrix[:15, :],
    )
    predicted_corrections, uncs = acsc.predict(p_structures[15:],)
    assert uncs is None
    assert predicted_corrections is not None
