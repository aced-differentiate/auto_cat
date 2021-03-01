"""Unit tests for the `autocat.learning.featurizersi` module."""

import os
import numpy as np

import pytest

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import CoulombMatrix

import qml

from autocat.io.qml import ase_atoms_to_qml_compound

from autocat.surface import generate_surface_structures
from autocat.learning.featurizers import full_structure_featurization


def test_full_structure_featurization_sine():
    # Tests Sine Matrix Generation
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    sine_matrix = full_structure_featurization(surf)
    sm = SineMatrix(n_atoms_max=len(surf), permutation="none")
    assert np.allclose(sine_matrix, sm.create(surf))
    # Check padding
    sine_matrix = full_structure_featurization(surf, size=40)
    assert sine_matrix.shape == (1600,)


def test_full_structure_featurization_coulomb():
    # Tests Coulomb Matrix Generation
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    coulomb_matrix = full_structure_featurization(surf, featurizer="coulomb_matrix")
    cm = CoulombMatrix(n_atoms_max=len(surf), permutation="none")
    assert np.allclose(coulomb_matrix, cm.create(surf))
    # Check padding
    coulomb_matrix = full_structure_featurization(surf, size=45)
    assert coulomb_matrix.shape == (2025,)


def test_full_structure_featurization_bob():
    # Tests Bag of Bonds generation
    surf = generate_surface_structures(["Ru"])["Ru"]["hcp0001"]["structure"]
    bob = full_structure_featurization(surf, featurizer="bob")
    qml_struct = ase_atoms_to_qml_compound(surf)
    qml_struct.generate_bob()
    assert np.allclose(bob, qml_struct.representation)
