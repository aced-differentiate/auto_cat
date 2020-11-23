"""Unit tests for the `autocat.adsorption` module."""

import os
import shutil

import pytest
from pytest import approx
from pytest import raises

from ase.build import molecule
from autocat.adsorption import generate_rxn_structures
from autocat.adsorption import generate_molecule_object
from autocat.intermediates import *
from autocat.surface import generate_surface_structures


def test_generate_rxn_structures_adsorbates():
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc111"]["structure"]
    # Test default
    ads = generate_rxn_structures(surf)
    assert "H" in ads
    # Test manual specification
    ads = generate_rxn_structures(surf, ads=["O2", "NH2"])
    with raises(KeyError):
        ads["H"]
    assert "O2" in ads
    # Test that reaction presets are used
    ads = generate_rxn_structures(surf, ads=orr_intermediate_names)
    assert list(ads.keys()) == ["OOH", "O", "OH"]


def test_generate_rxn_structures_references():
    # Test generation of reference states
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc111"]["structure"]
    ads = generate_rxn_structures(surf, ads=["NH"], refs=["N2", "H2", "NH3"])
    assert list(ads["references"].keys()) == ["N2", "H2", "NH3"]


def test_generate_rxn_structures_manualsites():
    # Test manual specification of adsorption sites
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_rxn_structures(
        surf,
        ads=["O"],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)], "custom": [(0.5, 0.5)]},
    )
    assert ads["O"]["origin"]["0.0_0.0"]["structure"][-1].symbol == "O"
    assert ads["O"]["origin"]["0.0_0.0"]["structure"][-1].x == approx(0.0)
    assert ads["O"]["custom"]["0.5_0.5"]["structure"][-1].y == approx(0.5)


def test_generate_rxn_structures_atoms_object():
    # Tests giving an Atoms object instead of a str as the adsorbate to be placed
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    m = molecule("CO")
    ads = generate_rxn_structures(
        surf,
        ads=[m, "H"],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)]},
        mol_indices={m.get_chemical_formula(): 1},
    )
    assert "CO" in ads
    assert len(ads["CO"]["origin"]["0.0_0.0"]["structure"]) == (len(surf) + len(m))
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-1].symbol == "C"
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-2].symbol == "O"


def test_generate_rxn_structures_autosites():
    # Test automated placement of adsorbate
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_rxn_structures(
        surf, ads=["H"], all_sym_sites=True, site_type=["ontop", "hollow"]
    )
    assert list(ads["H"].keys()) == ["ontop", "hollow"]
    assert list(ads["H"]["ontop"].keys()) == ["0.0_0.0"]
    # Test all sites automatically identified
    ads = generate_rxn_structures(surf, ads=["H"], all_sym_sites=True)
    assert len(ads["H"]["ontop"]) == 1
    assert len(ads["H"]["hollow"]) == 2
    assert len(ads["H"]["bridge"]) == 1
