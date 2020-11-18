"""Unit tests for the `autocat.saa` module."""

import os
import shutil

import pytest
from pytest import approx
from pytest import raises

import numpy as np
from autocat.saa import generate_saa_structures


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
