"""Unit tests for the `autocat.io.rmg` module."""

import ase
import numpy as np
import os

import pytest

from ase import Atoms
from rmgpy.molecule.molecule import Molecule

from autocat.io.rmg import rmgmol_to_ase_atoms_list
from autocat.io.rmg import output_yaml_to_surface_rmg_mol
from autocat.io.rmg import _remove_x


def test_rmgmol_to_ase_atoms_list():
    # Tests converting an rmg molecule object to an ase Atoms object
    rmgmol = Molecule().from_smiles("O=C=O")
    assert isinstance(rmgmol, Molecule)
    ase_atoms_list = rmgmol_to_ase_atoms_list(rmgmol)
    assert isinstance(ase_atoms_list[0], Atoms)
    assert ase_atoms_list[0].get_chemical_formula() == "CO2"
    # Test returning multiple conformers
    ase_atoms_list = rmgmol_to_ase_atoms_list(
        rmgmol, num_conformers=3, return_only_lowest_energy_conformer=False
    )
    assert len(ase_atoms_list) == 3
    assert isinstance(ase_atoms_list[-1], Atoms)
    # Test without optimization
    rmgmol = Molecule().from_smiles("O")
    ase_atoms_list = rmgmol_to_ase_atoms_list(rmgmol, optimize=False, num_conformers=2)
    assert len(ase_atoms_list) == 1
    assert isinstance(ase_atoms_list[0], Atoms)
    assert ase_atoms_list[0].get_chemical_formula() == "H2O"


def test_remove_x():
    # Tests removing X from an rmgmol object
    adj_list = "1 O u0 p2 c0 {2,D}\n2 X u0 p0 c0 {1,D}\n"
    rmgmol = Molecule().from_adjacency_list(adj_list)
    rm_rmgmol = _remove_x(rmgmol)
    assert not rm_rmgmol.contains_surface_site()
    rmgmol = Molecule().from_smiles("C")
    assert rmgmol == _remove_x(rmgmol)
    assert not _remove_x(rmgmol).contains_surface_site()
    # Tests with multiple Xs
    rmgmol = Molecule().from_adjacency_list(
        "1 O u0 p2 c0 {2,S} {5,S}\n\
        2 C u0 p0 c0 {1,S} {3,S} {4,D}\n\
        3 H u0 p0 c0 {2,S}\n\
        4 X u0 p0 c0 {2,D}\n\
        5 X u0 p0 c0 {1,S}\n"
    )
    rm_rmgmol = _remove_x(rmgmol)
    assert not rm_rmgmol.contains_surface_site()
