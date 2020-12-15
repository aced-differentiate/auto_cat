"""Unit tests for the `autocat.mpea` module."""

import os
import shutil

import pytest
from pytest import approx
from pytest import raises

from autocat.mpea import generate_mpea_random
from autocat.mpea import random_population


def test_random_population_lattice():
    # Test that lattice parameter is average
    mpea = random_population(["Pt", "Fe", "Cu"], supcell=(1, 1, 3))
    assert mpea.cell[0][0] == approx(2.4513035081)
    # Test that lattice parameter is weighted based on composition
    mpea = random_population(
        ["Pt", "Fe", "Cu"], composition={"Pt": 2}, supcell=(1, 1, 3)
    )
    assert mpea.cell[0][0] == approx(2.531442276)
    # Test that lattice parameters are pulled from the correct default library
    mpea = random_population(
        ["Pt", "Pd", "Fe"], supcell=(1, 1, 3), default_lattice_library="beefvdw_fd"
    )
    assert mpea.cell[0][0] == approx(2.57036143)


def test_generate_mpea_random_samples():
    # Test that the number of samples specified are generated
    mpeas = generate_mpea_random(["Pt", "Pd", "Ir", "Cu"], num_of_samples=20)
    assert len(list(mpeas["fcc110"].keys())) == 20


def test_generate_mpea_random_facets():
    # Test default facets given a crystal structure
    mpeas = generate_mpea_random(["Pt", "Pd", "Ir", "Cu"], crystal_structure="fcc")
    with raises(KeyError):
        m = mpeas["bcc100"]
    assert list(mpeas.keys()) == ["fcc100", "fcc111", "fcc110"]
    mpeas = generate_mpea_random(["Pt", "Pd", "Ir", "Cu"], crystal_structure="bcc")
    assert list(mpeas.keys()) == ["bcc100", "bcc111", "bcc110"]
    # Test manually specified facets
    mpeas = generate_mpea_random(
        ["Pt", "Pd", "Ir", "Cu"], crystal_structure="bcc", facets=["111"]
    )
    with raises(KeyError):
        m = mpeas["bcc110"]


def test_generate_mpea_random_write_location():
    # Test user-specified write location
    mpeas = generate_mpea_random(
        ["Pt", "Pd", "Ir", "Cu"], write_to_disk=True, write_location="test_dir"
    )
    assert os.path.samefile(
        mpeas["fcc111"]["1"]["traj_file_path"],
        "test_dir/Pt1.0Pd1.0Ir1.0Cu1.0/fcc111/1/structure/input.traj",
    )
    shutil.rmtree("test_dir")


def test_generate_mpea_random_dirs_exist_ok():
    mpeas = generate_mpea_random(["Pt", "Pd", "Ir", "Cu"], write_to_disk=True)
    with raises(FileExistsError):
        mpeas = generate_mpea_random(["Pt", "Pd", "Ir", "Cu"], write_to_disk=True)
    mpeas = generate_mpea_random(
        ["Pt", "Pd", "Ir", "Cu"], write_to_disk=True, dirs_exist_ok=True
    )
    assert os.path.samefile(
        mpeas["fcc100"]["5"]["traj_file_path"],
        "Pt1.0Pd1.0Ir1.0Cu1.0/fcc100/5/structure/input.traj",
    )
    shutil.rmtree("Pt1.0Pd1.0Ir1.0Cu1.0")
