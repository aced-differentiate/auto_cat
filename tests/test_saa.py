"""Unit tests for the `autocat.saa` module."""

import os
import shutil

import pytest
from pytest import approx
from pytest import raises

import numpy as np
from autocat.saa import generate_saa_structures
from autocat.saa import generate_doped_structures
from autocat.saa import _find_sa_ind
from autocat.surface import generate_surface_structures


def test_generate_saa_structures_host_species_list():
    saa = generate_saa_structures(["Pt", "Cu"], ["Fe"])
    assert "Pt" in saa
    assert "Cu" in saa


def test_generate_saa_structures_dops_species_list():
    # Test dopants added to each host
    saa = generate_saa_structures(["Pt", "Cu"], ["Fe"])
    assert "Fe" in saa["Pt"]
    assert "Fe" in saa["Cu"]
    # Test dopant species not added to itself
    saa = generate_saa_structures(["Pt", "Cu"], ["Fe", "Cu"])
    assert "Cu" not in saa["Cu"]


def test_generate_saa_structures_sa_mag_defaults():
    # Test default magnetic moment given to SA
    saa = generate_saa_structures(["Cu"], ["Fe", "Ni"])
    st = saa["Cu"]["Fe"]["fcc111"]["structure"]
    assert st[_find_sa_ind(st, "Fe")].magmom == approx(4.0)
    st = saa["Cu"]["Ni"]["fcc111"]["structure"]
    assert st[_find_sa_ind(st, "Ni")].magmom == approx(2.0)


def test_generate_saa_structures_write_location():
    # Test user-specified write location
    saa = generate_saa_structures(
        ["Pt", "Cu"], ["Fe"], write_to_disk=True, write_location="test_dir"
    )
    assert os.path.samefile(
        saa["Pt"]["Fe"]["fcc111"]["traj_file_path"], "test_dir/Pt/Fe/fcc111/input.traj"
    )
    assert os.path.samefile(
        saa["Cu"]["Fe"]["fcc100"]["traj_file_path"], "test_dir/Cu/Fe/fcc100/input.traj"
    )
    shutil.rmtree("test_dir")


def test_generate_saa_structures_dirs_exist_ok():
    saa = generate_saa_structures(["Pt", "Cu"], ["Fe"], write_to_disk=True)
    with raises(FileExistsError):
        saa = generate_saa_structures(["Pt", "Cu"], ["Fe"], write_to_disk=True)
    saa = generate_saa_structures(
        ["Pt", "Cu"], ["Fe"], write_to_disk=True, dirs_exist_ok=True
    )
    assert os.path.samefile(
        saa["Pt"]["Fe"]["fcc110"]["traj_file_path"], "Pt/Fe/fcc110/input.traj"
    )
    shutil.rmtree("Pt")
    shutil.rmtree("Cu")


def test_generate_doped_structures_fix_layers():
    # Test layers remain fixed after doping
    host = generate_surface_structures(["Pt"], fix=2, supcell=(3, 3, 4))["Pt"][
        "fcc111"
    ]["structure"]
    dop_host = generate_doped_structures(host, "Fe")["27"]["structure"]
    assert (dop_host.constraints[0].get_indices() == np.arange(0, 18)).any()
    assert dop_host.constraints[0].todict()["name"] == "FixAtoms"


def test_generate_doped_structures_keep_host_mag():
    # Test that host magnetization is kept after doping
    host = generate_surface_structures(["Fe"], fix=2, supcell=(3, 3, 4))["Fe"][
        "bcc111"
    ]["structure"]
    dop_host = generate_doped_structures(host, "Ni", dopant_magnetic_moment=2.0)["27"][
        "structure"
    ]
    assert 4.0 in dop_host.get_initial_magnetic_moments()
    assert 0.0 not in dop_host.get_initial_magnetic_moments()
