"""Unit tests for the `autocat.io.rmg` module."""

import numpy as np
import os

import pytest

from ase import Atoms
from rmgpy.molecule.converter import to_rdkit_mol
from rmgpy.molecule.molecule import Molecule
from rmgpy.chemkin import load_species_dictionary
from rdkit.Chem.rdmolfiles import SDWriter

from autocat.io.rmg import rmgmol_to_ase_atoms
from autocat.io.rmg import output_yaml_to_surface_rmg_mol


def test_rmgmol_to_ase_atoms():
    # Tests converting an rmg molecule object to an ase Atoms object
    rmgmol = Molecule().from_smiles("O=C=O")
    assert isinstance(rmgmol, Molecule)
    ase_atoms = rmgmol_to_ase_atoms(rmgmol)
    assert isinstance(ase_atoms, Atoms)
    assert ase_atoms.get_chemical_formula() == "CO2"
