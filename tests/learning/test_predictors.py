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
from autocat.learning.predictors import AutoCatStructureCorrector


def test_fit_model_on_perturbed_systems():
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
    acsc = AutoCatStructureCorrector(
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6}
    )
    acsc.fit(
        p_structures,
        adsorbate_indices_dictionary={
            base_struct.get_chemical_formula() + "_" + str(i): [-1, -2]
            for i in range(15)
        },
        collected_matrices=collected_matrices,
    )
    assert acsc.adsorbate_featurizer == "soap"
    assert acsc.is_fit

    # check no longer fit after changing setting
    acsc.adsorbate_featurizer = "acsf"
    assert acsc.adsorbate_featurizer == "acsf"
    assert not acsc.is_fit


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
    acsc = AutoCatStructureCorrector(
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6}
    )
    acsc.fit(
        p_structures[:15],
        adsorbate_indices_dictionary={
            base_struct.get_chemical_formula() + "_" + str(i): [-1, -2]
            for i in range(15)
        },
        collected_matrices=collected_matrices[:15, :],
    )
    predicted_correction_matrix, corrected_structures = acsc.predict(
        p_structures[15:],
        adsorbate_indices_dictionary={
            base_struct.get_chemical_formula() + "_" + str(i): [-1, -2]
            for i in range(5)
        },
    )
    assert isinstance(corrected_structures[0], Atoms)
    assert len(corrected_structures) == 5
    assert isinstance(predicted_correction_matrix, np.ndarray)
    assert predicted_correction_matrix.shape[0] == 5
