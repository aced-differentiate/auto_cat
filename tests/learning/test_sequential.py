"""Unit tests for the `autocat.learning.sequential` module"""

import os
import pytest
import numpy as np

from autocat.learning.predictors import AutoCatStructureCorrector
from autocat.learning.sequential import simulated_sequential_learning
from autocat.surface import generate_surface_structures
from autocat.adsorption import place_adsorbate


def test_simulated_sequential_outputs():
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