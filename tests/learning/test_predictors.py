"""Unit tests for the `autocat.learning.predictors` module"""

import os
import pytest
import numpy as np

from sklearn.kernel_ridge import KernelRidge

from ase import Atoms

from autocat.adsorption import place_adsorbate
from autocat.surface import generate_surface_structures
from autocat.perturbations import generate_perturbed_dataset
from autocat.learning.featurizers import get_X
from autocat.learning.featurizers import catalyst_featurization
from autocat.learning.predictors import get_trained_model_on_perturbed_systems
from autocat.learning.predictors import predict_initial_configuration


def test_get_trained_model_on_perturbed_systems():
    # Test returns a fit model
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "OH")["custom"]["structure"]
    p_set = generate_perturbed_dataset(
        [base_struct],
        atom_indices_to_perturb_dictionary={
            base_struct.get_chemical_formula() + "_0": [-1, -2]
        },
        num_of_perturbations=15,
    )
    p_structures = p_set["collected_structures"]
    collected_matrices = p_set["collected_matrices"]
    trained_model = get_trained_model_on_perturbed_systems(
        p_structures,
        adsorbate_indices_dictionary={
            base_struct.get_chemical_formula() + "_" + str(i): [-1, -2]
            for i in range(15)
        },
        collected_matrices=collected_matrices,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    # check correct regressor is used
    assert isinstance(trained_model, KernelRidge)

    # check if fit
    t_feat = catalyst_featurization(
        p_structures[0],
        [-1, -2],
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    # will raise a NotFittedError if not fit
    trained_model.predict(t_feat.reshape(1, -1))


def test_predict_initial_configuration_formats():
    # Test outputs are returned as expected
    sub = generate_surface_structures(["Fe"], facets={"Fe": ["100"]})["Fe"]["bcc100"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "CO")["custom"]["structure"]
    p_set = generate_perturbed_dataset(
        [base_struct],
        atom_indices_to_perturb_dictionary={
            base_struct.get_chemical_formula() + "_0": [-1, -2]
        },
        num_of_perturbations=20,
    )
    p_structures = p_set["collected_structures"]
    collected_matrices = p_set["collected_matrices"]
    trained_model = get_trained_model_on_perturbed_systems(
        p_structures,
        adsorbate_indices_dictionary={
            base_struct.get_chemical_formula() + "_" + str(i): [-1, -2]
            for i in range(20)
        },
        collected_matrices=collected_matrices,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    predicted_correction_matrix, corrected_structure = predict_initial_configuration(
        p_structures[0],
        [-1, -2],
        trained_model,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    assert isinstance(corrected_structure, Atoms)
    assert isinstance(predicted_correction_matrix, np.ndarray)
    # check correction matrix returned in shape of coordinates matrix
    assert predicted_correction_matrix.shape == (len(corrected_structure), 3)
