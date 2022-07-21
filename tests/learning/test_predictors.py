"""Unit tests for the `autocat.learning.predictors` module"""
import os

import pytest
import numpy as np
import tempfile

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from dscribe.descriptors import SOAP

from autocat.adsorption import generate_adsorbed_structures
from autocat.surface import generate_surface_structures
from autocat.learning.featurizers import Featurizer
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
    featurizer = Featurizer(
        featurizer_class=SOAP,
        species_list=["Pt", "Fe", "Ru", "O", "H"],
        kwargs={"rcut": 6.0, "nmax": 6, "lmax": 6},
    )
    regressor = GaussianProcessRegressor()
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
    acsc.fit(
        training_structures=structs, y=labels,
    )
    assert acsc.is_fit
    assert check_is_fitted(acsc.regressor) is None

    # check no longer fit after changing featurizer
    featurizer.species_list = ["Pt", "Fe", "Ru", "O", "H", "N"]
    featurizer.kwargs = {"rcut": 7.0, "nmax": 8, "lmax": 8}
    acsc.featurizer = featurizer
    assert not acsc.is_fit

    acsc.fit(
        training_structures=structs, y=labels,
    )
    assert acsc.is_fit

    # check no longer fit after changing model class
    regressor = KernelRidge()
    acsc.regressor = regressor
    assert not acsc.is_fit


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
    featurizer = Featurizer(
        featurizer_class=SOAP,
        species_list=["Pt", "Fe", "Ru", "O", "H"],
        kwargs={"rcut": 6.0, "nmax": 6, "lmax": 6},
    )
    regressor = GaussianProcessRegressor()
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
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
    acsc.regressor = KernelRidge()
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
    featurizer = Featurizer(
        featurizer_class=SOAP,
        species_list=["Pt", "Fe", "Ru", "O", "H"],
        kwargs={"rcut": 6.0, "nmax": 6, "lmax": 6},
    )
    regressor = GaussianProcessRegressor()
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
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


def test_predictor_from_json():
    # Tests generation of a Predictor from a json
    featurizer = Featurizer(
        featurizer_class=SOAP,
        species_list=["Pt", "Fe", "Ru", "O", "H"],
        kwargs={"rcut": 6.0, "nmax": 6, "lmax": 6},
    )
    regressor = RandomForestRegressor(25)
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsc.write_json_to_disk(write_location=_tmp_dir, json_name="testing_pred.json")
        json_path = os.path.join(_tmp_dir, "testing_pred.json")
        written_pred = Predictor.from_json(json_path)
        assert written_pred.regressor.get_params() == regressor.get_params()
        assert written_pred.featurizer == featurizer

    # check defaults
    regressor = RandomForestRegressor()
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsc.write_json_to_disk(write_location=_tmp_dir, json_name="testing_pred.json")
        json_path = os.path.join(_tmp_dir, "testing_pred.json")
        written_pred = Predictor.from_json(json_path)
        assert written_pred.regressor.get_params() == regressor.get_params()
        assert written_pred.featurizer == featurizer

    # does not save kwargs when not immediately serializable
    regressor = GaussianProcessRegressor(kernel=RBF(1.5))
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
    with tempfile.TemporaryDirectory() as _tmp_dir:
        acsc.write_json_to_disk(write_location=_tmp_dir, json_name="testing_pred.json")
        json_path = os.path.join(_tmp_dir, "testing_pred.json")
        written_pred = Predictor.from_json(json_path)
        assert (
            written_pred.regressor.get_params()
            == GaussianProcessRegressor().get_params()
        )
        assert written_pred.featurizer == featurizer


def test_predictor_to_jsonified_dict():
    # Tests converting a Predictor to a jsonified dict
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
    featurizer = Featurizer(
        featurizer_class=SOAP,
        species_list=["Pt", "Fe", "Ru", "O", "H"],
        kwargs={"rcut": 6.0, "nmax": 6, "lmax": 6},
    )
    regressor = RandomForestRegressor(n_estimators=75)
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
    jsonified_dict = acsc.to_jsonified_dict()
    assert jsonified_dict["featurizer"]["featurizer_class"] == {
        "module_string": "dscribe.descriptors.soap",
        "class_string": "SOAP",
    }
    assert jsonified_dict["featurizer"]["species_list"] == ["Fe", "Ru", "Pt", "O", "H"]
    assert jsonified_dict["featurizer"]["kwargs"] == {"rcut": 6.0, "nmax": 6, "lmax": 6}
    assert jsonified_dict["regressor"] == {
        "module_string": "sklearn.ensemble._forest",
        "name_string": "RandomForestRegressor",
        "kwargs": regressor.get_params(),
    }

    # check defaults
    regressor = RandomForestRegressor()
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
    jsonified_dict = acsc.to_jsonified_dict()
    assert jsonified_dict["regressor"] == {
        "module_string": "sklearn.ensemble._forest",
        "name_string": "RandomForestRegressor",
        "kwargs": regressor.get_params(),
    }

    # check doesn't include kwargs since non-serializable
    regressor = GaussianProcessRegressor(kernel=RBF(1.5))
    acsc = Predictor(regressor=regressor, featurizer=featurizer)
    jsonified_dict = acsc.to_jsonified_dict()
    assert jsonified_dict["regressor"]["kwargs"] is None
