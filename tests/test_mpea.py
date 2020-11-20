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
