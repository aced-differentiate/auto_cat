"""Unit tests for the `autocat.bulk` module."""

import os
import shutil

from pytest import approx
from pytest import raises

from autocat.bulk import generate_bulk_structures


def test_generate_bulk_structures_species_list():
    bs = generate_bulk_structures(["Li", "C"])
    assert "Li" in bs
    assert "C" in bs


def test_generate_bulk_structures_defaults():
    # Test default crystal structures (lattice vectors)
    # Test default ferromagnetic moment initialization
    # Test default write options
    bs = generate_bulk_structures(["Li", "Fe", "Ni"])
    assert bs["Li"]["crystal_structure"].cell[0][0] == approx(-1.745)
    magmoms = bs["Fe"]["crystal_structure"].get_initial_magnetic_moments()
    assert magmoms == approx([4.0])
    assert bs["Li"]["traj_file_path"] is None


def test_generate_bulk_structures_crystal_structures():
    species_list = ["Li", "Fe"]
    crystal_structures_err = {"Li": "scc", "Fe": "hcp"}
    crystal_structures_ok = {"Fe": "bcc"}
    # Test lattice parameter not specified error from `ase.bulk`
    with raises(ValueError):
        bs = generate_bulk_structures(
            species_list, crystal_structures=crystal_structures_err
        )
    # Test dlattice parameter not specified error from `ase.bulk`
    bs = generate_bulk_structures(
        species_list, crystal_structures=crystal_structures_ok
    )
    assert bs["Fe"]["crystal_structure"].cell[0][0] == approx(-1.435)


def test_generate_bulk_structures_lattice_parameters():
    bs = generate_bulk_structures(
        ["Fe"],
        crystal_structures={"Fe": "hcp"},
        a_dict={"Fe": 3.2},
        c_dict={"Fe": 4.5},
    )
    assert bs["Fe"]["crystal_structure"].cell[0][0] == approx(3.2)
    assert bs["Fe"]["crystal_structure"].cell[2][2] == approx(4.5)


def test_generate_bulk_structures_ref_library():
    # Tests pulling lattice parameters from pbe_fd ref library
    bs = generate_bulk_structures(
        ["W", "Pd"], a_dict={"Pd": 3.94}, default_lattice_library="pbe_fd"
    )
    assert bs["W"]["crystal_structure"].cell[0][0] == approx(-1.590292)
    assert bs["Pd"]["crystal_structure"].cell[1][0] == approx(1.97)
    # Tests pulling lattice parameters from beefvdw_pw ref library
    bs = generate_bulk_structures(["Ru"], default_lattice_library="beefvdw_pw")
    assert bs["Ru"]["crystal_structure"].cell[0][0] == approx(2.738748)
    assert bs["Ru"]["crystal_structure"].cell[2][2] == approx(4.316834)


def test_generate_bulk_structures_magnetic_moments():
    # Test ground state magnetic moments from `ase.data`
    bs = generate_bulk_structures(["Cu"], set_magnetic_moments=["Cu"])
    magmoms = bs["Cu"]["crystal_structure"].get_initial_magnetic_moments()
    assert magmoms == approx([1.0])
    # Test user-specified magnetic moments
    bs = generate_bulk_structures(
        ["Cu"], set_magnetic_moments=["Cu"], magnetic_moments={"Cu": 2.1}
    )
    magmoms = bs["Cu"]["crystal_structure"].get_initial_magnetic_moments()
    assert magmoms == approx([2.1])


def test_generate_bulk_structures_write_location():
    # Test user-specified write location
    # TODO(@hegdevinayi): Use temporary file/directory from `tempfile` instead
    # of writing to disk. Example here:
    # https://github.com/CitrineInformatics/dft-input-gen/blob/master/tests/qe/test_pwx.py#L320
    bs = generate_bulk_structures(
        ["Li", "Ti"], write_to_disk=True, write_location="test_dir"
    )
    assert os.path.samefile(
        bs["Li"]["traj_file_path"], "test_dir/Li_bulk_bcc/input.traj"
    )
    shutil.rmtree("test_dir")


def test_generate_bulk_structures_dirs_exist_ok():
    bs = generate_bulk_structures(["Li"], write_to_disk=True)
    with raises(FileExistsError):
        bs = generate_bulk_structures(["Li"], write_to_disk=True)
    # Test no error on dirs_exist_ok = True, and check default file path
    bs = generate_bulk_structures(["Li"], write_to_disk=True, dirs_exist_ok=True)
    assert os.path.samefile(bs["Li"]["traj_file_path"], "Li_bulk_bcc/input.traj")
    shutil.rmtree(os.path.dirname(bs["Li"]["traj_file_path"]))
