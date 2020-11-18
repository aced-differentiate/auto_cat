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
    surf = generate_surface_structures(["Pt"], supcell=(3, 3, 4), fix=2)
    assert (
        surf["Pt"]["fcc111"]["structure"].constraints[0].get_indices()
        == np.arange(0, 18)
    ).any()
    assert (
        surf["Pt"]["fcc111"]["structure"].constraints[0].todict()["name"] == "FixAtoms"
    )


def test_generate_surface_structures_vacuum():
    # Test vacuum size
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


def test_generate_surface_structures_cell():
    # Test default a
    surf = generate_surface_structures(["Fe"], supcell=(1, 1, 3))
    assert surf["Fe"]["bcc110"]["structure"].cell[0][0] == approx(2.87)
    # Test a_dict
    surf = generate_surface_structures(["Pt"], a_dict={"Pt": 3.95}, supcell=(1, 1, 3))
    assert surf["Pt"]["fcc111"]["structure"].cell[0][0] == approx(2.793071785)
    # Test default c
    surf = generate_surface_structures(["Zn"], supcell=(1, 1, 3), vac=10.0)
    assert (surf["Zn"]["hcp0001"]["structure"].cell[2][2] - 20.0) == approx(4.93696)
    # Test c_dict
    surf = generate_surface_structures(
        ["Ru"], c_dict={"Ru": 4.35}, supcell=(1, 1, 3), vac=10.0
    )
    assert (surf["Ru"]["hcp0001"]["structure"].cell[2][2] - 20.0) == approx(4.35)


def test_generate_surface_structures_supcell():
    # Test supercell size in the xy plane
    surf1 = generate_surface_structures(["Pt"], supcell=(1, 1, 3))
    surf2 = generate_surface_structures(["Pt"], supcell=(2, 2, 3))
    assert surf2["Pt"]["fcc111"]["structure"].cell[0][0] == approx(
        2.0 * surf1["Pt"]["fcc111"]["structure"].cell[0][0]
    )
    assert surf2["Pt"]["fcc111"]["structure"].cell[1][1] == approx(
        2.0 * surf1["Pt"]["fcc111"]["structure"].cell[1][1]
    )
    # Test supercell size in the z axis
    assert max(surf2["Pt"]["fcc100"]["structure"].get_tags()) == 3


def test_generate_surface_structures_write_location():
    # Test user-specified write location
    surf = generate_surface_structures(
        ["Au", "Ir"], write_to_disk=True, write_location="test_dir"
    )
    assert os.path.samefile(
        surf["Ir"]["fcc111"]["traj_file_path"], "test_dir/Ir/fcc111/input.traj"
    )
    shutil.rmtree("test_dir")


def test_generate_surface_structures_dirs_exist_ok():
    surf = generate_surface_structures(
        ["Au"], ft_dict={"Au": ["100"]}, write_to_disk=True
    )
    with raises(FileExistsError):
        surf = generate_surface_structures(
            ["Au"], ft_dict={"Au": ["100"]}, write_to_disk=True
        )
    # Test no error on dirs_exist_ok = True, and check default file path
    surf = generate_surface_structures(
        ["Au"], ft_dict={"Au": ["100"]}, write_to_disk=True, dirs_exist_ok=True
    )
    assert os.path.samefile(
        surf["Au"]["fcc100"]["traj_file_path"], "Au/fcc100/input.traj"
    )
    shutil.rmtree("Au")
