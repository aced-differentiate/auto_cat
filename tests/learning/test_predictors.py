"""Unit tests for the `autocat.learning.predictors` module"""

import pytest
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import SOAP

from matminer.featurizers.composition import ElementProperty

from ase import Atoms

from autocat.adsorption import generate_adsorbed_structures, place_adsorbate
from autocat.surface import generate_surface_structures
from autocat.learning.predictors import Predictor
from autocat.learning.predictors import PredictorError
from autocat.utils import flatten_structures_dict


def test_fit():
    # Test returns a fit model
    subs = flatten_structures_dict(generate_surface_structures(["Pt", "Fe", "Ru"]))
    structs = []
    for sub in subs:
        ads_struct = flatten_structures_dict(
            generate_adsorbed_structures(
                surface=sub,
                adsorbates=["OH"],
                adsorption_sites={"origin": [(0.0, 0.0)]},
                use_all_sites=False,
            )
        )[0]
        structs.append(ads_struct)
    labels = np.random.rand(len(structs))
    acsc = Predictor(
        featurizer_class=SOAP,
        featurization_kwargs={
            "species_list": ["Pt", "Fe", "Ru", "O", "H"],
            "kwargs": {"rcut": 6.0, "nmax": 6, "lmax": 6},
        },
        model_class=GaussianProcessRegressor,
    )
    acsc.fit(
        training_structures=structs, y=labels,
    )
    assert acsc.is_fit
    assert check_is_fitted(acsc.regressor) is None

    # check no longer fit after changing featurization kwargs
    acsc.featurization_kwargs = {
        "species_list": ["Pt", "Fe", "Ru", "O", "H", "N"],
        "kwargs": {"rcut": 7.0, "nmax": 8, "lmax": 8},
    }
    assert not acsc.is_fit
    with pytest.raises(NotFittedError):
        check_is_fitted(acsc.regressor)

    acsc.fit(
        training_structures=structs, y=labels,
    )

    # check no longer fit after changing featurization class
    acsc.featurizer_class = SineMatrix
    assert not acsc.is_fit
    with pytest.raises(NotFittedError):
        check_is_fitted(acsc.regressor)

    acsc.fit(
        training_structures=structs, y=labels,
    )

    # check no longer fit after changing model class
    acsc.model_class = KernelRidge
    assert not acsc.is_fit
    with pytest.raises(NotFittedError):
        check_is_fitted(acsc.regressor)

    acsc.fit(
        training_structures=structs, y=labels,
    )

    # check no longer fit after changing model kwargs
    kernel = RBF()
    acsc.model_kwargs = {"kernel": kernel}
    assert not acsc.is_fit
    with pytest.raises(NotFittedError):
        check_is_fitted(acsc.regressor)


def test_predict():
    # Test outputs are returned as expected
    subs = flatten_structures_dict(generate_surface_structures(["Pt", "Fe", "Ru"]))
    structs = []
    for sub in subs:
        ads_struct = flatten_structures_dict(
            generate_adsorbed_structures(
                surface=sub,
                adsorbates=["OH"],
                adsorption_sites={"origin": [(0.0, 0.0)]},
                use_all_sites=False,
            )
        )[0]
        structs.append(ads_struct)
    labels = np.random.rand(len(structs))
    acsc = Predictor(
        featurizer_class=SOAP,
        featurization_kwargs={
            "species_list": ["Pt", "Fe", "Ru", "O", "H"],
            "kwargs": {"rcut": 6.0, "nmax": 6, "lmax": 6},
        },
        model_class=GaussianProcessRegressor,
    )
    acsc.fit(
        training_structures=structs[:-3], y=labels[:-3],
    )
    pred, unc = acsc.predict([structs[-3]],)
    assert len(pred) == 1
    # check dimension of uncertainty estimates
    assert len(unc) == 1

    pred, unc = acsc.predict(structs[-3:],)
    assert len(pred) == 3
    # check dimension of uncertainty estimates
    assert len(unc) == 3

    # Test prediction on model without uncertainty
    acsc.model_class = KernelRidge
    acsc.fit(
        training_structures=structs[:-3], y=labels[:-3],
    )
    pred, unc = acsc.predict([structs[-2]],)
    assert len(pred) == 1
    assert unc is None


def test_score():
    # Tests the score method
    subs = flatten_structures_dict(generate_surface_structures(["Pt", "Fe", "Ru"]))
    structs = []
    for sub in subs:
        ads_struct = flatten_structures_dict(
            generate_adsorbed_structures(
                surface=sub,
                adsorbates=["OH"],
                adsorption_sites={"origin": [(0.0, 0.0)]},
                use_all_sites=False,
            )
        )[0]
        structs.append(ads_struct)
    labels = np.random.rand(len(structs))
    acsc = Predictor(
        featurizer_class=SOAP,
        featurization_kwargs={
            "species_list": ["Pt", "Fe", "Ru", "O", "H"],
            "kwargs": {"rcut": 6.0, "nmax": 6, "lmax": 6},
        },
        model_class=GaussianProcessRegressor,
    )
    acsc.fit(
        training_structures=structs[:-3], y=labels[:-3],
    )
    mae = acsc.score(structs[-3:], labels[-3:])
    assert isinstance(mae, float)
    mse = acsc.score(structs[-2:], labels[-2:], metric="mse")
    assert isinstance(mse, float)

    # Test returning predictions
    _, preds, uncs = acsc.score(structs[-2:], labels[-2:], return_predictions=True)
    assert len(preds) == 2
    assert len(uncs) == 2
    # check catches unknown metric
    with pytest.raises(PredictorError):
        acsc.score(structs, labels, metric="msd")


def test_class_and_kwargs_logic():
    # Tests providing regression model class and kwargs
    featurization_kwargs = {
        "species_list": ["Pt", "Fe", "Ru", "O", "H"],
        "kwargs": {"rcut": 6.0, "nmax": 6, "lmax": 6, "sparse": True},
    }
    acsc = Predictor(
        model_class=KernelRidge,
        model_kwargs={"gamma": 0.5},
        featurizer_class=SOAP,
        featurization_kwargs=featurization_kwargs,
    )
    assert isinstance(acsc.regressor, KernelRidge)
    # check that regressor created with correct kwargs
    assert acsc.regressor.gamma == 0.5
    assert acsc.model_kwargs == {"gamma": 0.5}
    assert acsc.featurization_kwargs == featurization_kwargs
    assert acsc.featurizer.featurization_object.sparse

    # check that model kwargs are removed when model class is changed
    acsc.model_class = GaussianProcessRegressor
    assert acsc.model_kwargs is None
    assert acsc.featurizer_class == SOAP
    assert acsc.featurization_kwargs == featurization_kwargs

    # check that regressor is updated when model kwargs updated
    acsc.model_kwargs = {"alpha": 5e-10}
    assert acsc.regressor.alpha == 5e-10

    # check that featurization kwargs removed when featurization class changed
    acsc.featurizer_class = ElementProperty
    assert acsc.featurization_kwargs is None

    # check that featurizer is updated when featurization kwargs updated
    acsc.featurization_kwargs = {"preset": "magpie"}
    assert "Electronegativity" in acsc.featurizer.featurization_object.features

    acsc.featurization_kwargs = {"preset": "matminer"}
    assert (
        "coefficient_of_linear_thermal_expansion"
        in acsc.featurizer.featurization_object.features
    )

    acsc.featurizer_class = SineMatrix
    acsc.featurization_kwargs = {"kwargs": {"flatten": False}}
    assert not acsc.featurizer.featurization_object.flatten
    acsc.featurization_kwargs = {"kwargs": {"flatten": True}}
    assert acsc.featurizer.featurization_object.flatten
