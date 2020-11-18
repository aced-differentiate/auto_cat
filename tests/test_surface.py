"""Unit tests for the `autocat.surface` module."""

import os
import shutil

import pytest
from pytest import approx
from pytest import raises

import numpy as np
from autocat.surface import generate_surface_structures


def test_generate_surface_structures_species_list():
    surf = generate_surface_structures(["Pt", "Fe"])
    assert "Pt" in surf
    assert "Fe" in surf


def test_generate_surface_structures_facets():
    surf = generate_surface_structures(
        ["Pt", "Fe", "Ru", "Pd"], ft_dict={"Pd": ["100"]}
    )
    # Test default facet selection
    assert list(surf["Ru"].keys()) == ["hcp0001"]
    assert list(surf["Fe"].keys()) == ["bcc100", "bcc111", "bcc110"]
    assert list(surf["Pt"].keys()) == ["fcc100", "fcc111", "fcc110"]
    # Test facet specification
    assert list(surf["Pd"].keys()) == ["fcc100"]


def test_generate_surface_structures_fix_layers():
    # Test fixing of layers of the slab
    surf = generate_surface_structures(["Pt"], fix=2)
    assert (
        surf["Pt"]["fcc111"]["structure"].constraints[0].get_indices()
        == np.arange(0, 18)
    ).any()
    assert (
        surf["Pt"]["fcc111"]["structure"].constraints[0].todict()["name"] == "FixAtoms"
    )


def test_generate_surface_structures_vacuum():
    surf = generate_surface_structures(["Fe"], ft_dict={"Fe": ["111"]}, vac=15.0)
    st = surf["Fe"]["bcc111"]["structure"]
    assert (
        st.cell[2][2] - (max(st.positions[:, 2]) - min(st.positions[:, 2]))
    ) / 2.0 == approx(15.0)
    surf = generate_surface_structures(["Ru"], vac=10.0)
    st = surf["Ru"]["hcp0001"]["structure"]
    assert (
        st.cell[2][2] - (max(st.positions[:, 2]) - min(st.positions[:, 2]))
    ) / 2.0 == approx(10.0)
