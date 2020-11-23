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


def test_generate_rxn_structures_mol_placement():
    # Tests default height
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_rxn_structures(
        surf, all_sym_sites=False, sites={"origin": [(0.0, 0.0)]}
    )
    assert (
        ads["H"]["origin"]["0.0_0.0"]["structure"][-1].z
        - ads["H"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(1.5)
    # Tests manually specifying height
    ads = generate_rxn_structures(
        surf,
        ads=["OH", "O"],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)]},
        height={"OH": 2.0},
    )
    assert (
        ads["OH"]["origin"]["0.0_0.0"]["structure"][-2].z
        - ads["OH"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(2.0)
    assert (
        ads["O"]["origin"]["0.0_0.0"]["structure"][-1].z
        - ads["O"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(1.5)
    # Tests manually specifying mol_indices
    m = molecule("CO")
    ads = generate_rxn_structures(
        surf,
        ads=[m],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)]},
        mol_indices={m.get_chemical_formula(): 1},
    )
    assert (
        ads["CO"]["origin"]["0.0_0.0"]["structure"][-1].z
        - ads["CO"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(1.5)


def test_generate_rxn_structures_mol_rotation():
    # Tests applied rotations to adsorbates
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_rxn_structures(
        surf,
        ads=["NH3", "CO"],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)]},
        rots={"NH3": [[180.0, "x"], [90.0, "z"]], "CO": [[180.0, "y"]]},
    )
    # Check orientation of NH3
    assert ads["NH3"]["origin"]["0.0_0.0"]["structure"][-2].x == approx(-0.469865)
    assert ads["NH3"]["origin"]["0.0_0.0"]["structure"][-2].y == approx(0.813831)
    assert ads["NH3"]["origin"]["0.0_0.0"]["structure"][-2].z == approx(18.67793617)
    # Check orientation of CO
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-2].z == approx(18.28963917)
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-1].z == approx(19.43997917)


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


def test_generate_rxn_structures_write_location():
    # Test user-specified write location
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_rxn_structures(
        surf,
        ads=["OH", "O"],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0), (0.5, 0.5)], "custom": [(0.3, 0.3)]},
        write_to_disk=True,
        write_location="test_dir",
    )
    assert os.path.samefile(
        ads["OH"]["origin"]["0.0_0.0"]["traj_file_path"],
        "test_dir/OH/origin/0.0_0.0/input.traj",
    )
    assert os.path.samefile(
        ads["O"]["origin"]["0.5_0.5"]["traj_file_path"],
        "test_dir/O/origin/0.5_0.5/input.traj",
    )
    assert os.path.samefile(
        ads["O"]["custom"]["0.3_0.3"]["traj_file_path"],
        "test_dir/O/custom/0.3_0.3/input.traj",
    )
    ads = generate_rxn_structures(
        surf,
        ads=["OH"],
        site_type=["bridge"],
        write_to_disk=True,
        write_location="test_dir",
    )
    assert os.path.samefile(
        ads["OH"]["bridge"]["7.623_6.001"]["traj_file_path"],
        "test_dir/OH/bridge/7.623_6.001/input.traj",
    )
    shutil.rmtree("test_dir")


def test_generate_rxn_structure_dirs_exist_ok():
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_rxn_structures(
        surf, all_sym_sites=False, sites={"origin": [(0.0, 0.0)]}, write_to_disk=True
    )
    with raises(FileExistsError):
        ads = generate_rxn_structures(
            surf,
            all_sym_sites=False,
            sites={"origin": [(0.0, 0.0)]},
            write_to_disk=True,
        )
    ads = generate_rxn_structures(
        surf,
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)]},
        write_to_disk=True,
        dirs_exist_ok=True,
    )
    assert os.path.samefile(
        ads["H"]["origin"]["0.0_0.0"]["traj_file_path"], "H/origin/0.0_0.0/input.traj"
    )
    shutil.rmtree("H")
