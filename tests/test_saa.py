"""Unit tests for the `autocat.saa` module."""

import os
import tempfile

from pytest import approx
from pytest import raises

import numpy as np
from autocat.saa import generate_saa_structures
from autocat.saa import substitute_single_atom_on_surface
from autocat.saa import _find_dopant_index
from autocat.saa import _find_all_surface_atom_indices
from autocat.saa import AutocatSaaGenerationError
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
    assert st[_find_dopant_index(st, "Fe")].magmom == approx(4.0)
    st = saa["Cu"]["Ni"]["fcc111"]["structure"]
    assert st[_find_dopant_index(st, "Ni")].magmom == approx(2.0)


def test_generate_saa_structures_write_location():
    # Test user-specified write location
    _tmp_dir = tempfile.TemporaryDirectory().name
    saa = generate_saa_structures(
        ["Pt", "Cu"], ["Fe"], write_to_disk=True, write_location=_tmp_dir
    )
    assert os.path.samefile(
        saa["Pt"]["Fe"]["fcc111"]["traj_file_path"],
        os.path.join(_tmp_dir, "Pt", "Fe", "fcc111", "substrate", "input.traj"),
    )
    assert os.path.samefile(
        saa["Cu"]["Fe"]["fcc100"]["traj_file_path"],
        os.path.join(_tmp_dir, "Cu", "Fe", "fcc100", "substrate", "input.traj"),
    )


def test_generate_saa_structures_dirs_exist_ok():
    _tmp_dir = tempfile.TemporaryDirectory().name
    saa = generate_saa_structures(
        ["Pt", "Cu"], ["Fe"], write_to_disk=True, write_location=_tmp_dir
    )
    with raises(FileExistsError):
        saa = generate_saa_structures(
            ["Pt", "Cu"], ["Fe"], write_to_disk=True, write_location=_tmp_dir
        )
    saa = generate_saa_structures(
        ["Pt", "Cu"],
        ["Fe"],
        write_to_disk=True,
        write_location=_tmp_dir,
        dirs_exist_ok=True,
    )
    assert os.path.samefile(
        saa["Pt"]["Fe"]["fcc110"]["traj_file_path"],
        os.path.join(_tmp_dir, "Pt", "Fe", "fcc110", "substrate", "input.traj"),
    )


def test_substitute_single_atom_on_surface_fix_layers():
    # Test layers remain fixed after doping
    host = generate_surface_structures(
        ["Pt"], n_fixed_layers=2, supercell_dim=(3, 3, 4)
    )["Pt"]["fcc111"]["structure"]
    dop_host = substitute_single_atom_on_surface(host, "Fe")
    assert (dop_host.constraints[0].get_indices() == np.arange(0, 18)).any()
    assert dop_host.constraints[0].todict()["name"] == "FixAtoms"


def test_substitute_single_atom_on_surface_keep_host_mag():
    # Test that host magnetization is kept after doping
    host = generate_surface_structures(["Fe"])["Fe"]["bcc111"]["structure"]
    dop_host = substitute_single_atom_on_surface(host, "Ni", dopant_magnetic_moment=2.0)
    assert 4.0 in dop_host.get_initial_magnetic_moments()
    assert 0.0 not in dop_host.get_initial_magnetic_moments()


def test_substitute_single_atom_on_surface_cent_sa():
    # Test that dopant becomes centered within the cell
    host = generate_surface_structures(["Fe"])["Fe"]["bcc111"]["structure"]
    dop_host = substitute_single_atom_on_surface(
        host, "Ni", place_dopant_at_center=True
    )
    x = (dop_host.cell[0][0] + dop_host.cell[1][0]) / 2.0
    y = (dop_host.cell[0][1] + dop_host.cell[1][1]) / 2.0
    assert dop_host[_find_dopant_index(dop_host, "Ni")].x == approx(x)
    assert dop_host[_find_dopant_index(dop_host, "Ni")].y == approx(y)


def test_substitute_single_atom_on_surface_multi_unique_sites_to_dope():
    # Test catches system with multiple unique sites to dope
    host = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    host[19].symbol = "Au"
    with raises(NotImplementedError):
        substitute_single_atom_on_surface(host, "Ni", place_dopant_at_center=True)


def test_find_dopant_index():
    # Test helper function for finding dopant index
    # no dopant
    host = generate_surface_structures(["Cu"])["Cu"]["fcc111"]["structure"]
    with raises(AutocatSaaGenerationError):
        _find_dopant_index(host, "Au")
    # single dopant
    host[27].symbol = "Au"
    assert _find_dopant_index(host, "Au") == 27
    # multiple dopants
    host[32].symbol = "Au"
    with raises(NotImplementedError):
        _find_dopant_index(host, "Au")


def test_find_all_surface_atom_indices():
    # Test helper function for finding all surface atoms
    # clean elemental surface
    ru = generate_surface_structures(["Ru"], supercell_dim=(2, 2, 4))["Ru"]["hcp0001"][
        "structure"
    ]
    indices = _find_all_surface_atom_indices(ru)
    assert indices == [12, 13, 14, 15]

    pt110 = generate_surface_structures(["Pt"], supercell_dim=(1, 1, 4))["Pt"][
        "fcc110"
    ]["structure"]
    indices = _find_all_surface_atom_indices(pt110)
    assert indices == [3]

    # check increasing tolerance
    indices = _find_all_surface_atom_indices(pt110, tol=1.4)
    assert indices == [2, 3]

    pt100 = generate_surface_structures(["Pt"], supercell_dim=(3, 3, 4))["Pt"][
        "fcc100"
    ]["structure"]
    pt100[27].z += 0.3
    pt100[30].z -= 0.4
    indices = _find_all_surface_atom_indices(pt100, tol=0.6)
    assert indices == [27, 28, 29, 31, 32, 33, 34, 35]
