"""Unit tests for the `autocat.learning.featurizersi` module."""

import os
import numpy as np

import pytest

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

import qml

from autocat.io.qml import ase_atoms_to_qml_compound
from autocat.adsorption import generate_rxn_structures
from autocat.surface import generate_surface_structures
from autocat.learning.featurizers import full_structure_featurization
from autocat.learning.featurizers import adsorbate_featurization
from autocat.learning.featurizers import catalyst_featurization


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


def test_adsorbate_featurization_acsf():
    # Tests Atom Centered Symmetry Function generation
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    acsf_feat = adsorbate_featurization(ads_struct, [36])
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    acsf = ACSF(rcut=6.0, species=species)
    assert np.allclose(acsf_feat, acsf.create(ads_struct, [36]))


def test_adsorbate_featurization_soap():
    # Tests Smooth Overlap of Atomic Positions
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    soap_feat = adsorbate_featurization(
        ads_struct, [36], featurizer="soap", nmax=8, lmax=6
    )
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    soap = SOAP(rcut=6.0, species=species, nmax=8, lmax=6)
    assert np.allclose(soap_feat, soap.create(ads_struct, [36]))


def test_catalyst_featurization_concatentation():
    # Tests that the representations are properly concatenated
    # with kwargs input appropriately
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    cat = catalyst_featurization(
        ads_struct,
        [36],
        adsorbate_featurization_kwargs={"rcut": 5.0},
        structure_featurization_kwargs={"size": 40},
    )
    sm = SineMatrix(n_atoms_max=40, permutation="none")
    struct = sm.create(ads_struct).reshape(-1,)
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    acsf = ACSF(rcut=5.0, species=species)
    ads = acsf.create(ads_struct, [36]).reshape(-1,)
    cat_ref = np.concatenate((struct, ads))
    assert np.allclose(cat, cat_ref)
