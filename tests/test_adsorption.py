"""Unit tests for the `autocat.adsorption` module."""

import os
import shutil
import tempfile

import pytest
from pytest import approx
from pytest import raises

from ase.build import molecule

from autocat.saa import generate_saa_structures
from autocat.surface import generate_surface_structures
from autocat.adsorption import generate_adsorbed_structures
from autocat.adsorption import generate_molecule
from autocat.adsorption import get_adsorbate_height_estimate
from autocat.adsorption import get_adsorbate_slab_nn_list
from autocat.adsorption import place_adsorbate
from autocat.adsorption import AutocatAdsorptionGenerationError


def test_generate_molecule_from_name_error():
    with pytest.raises(AutocatAdsorptionGenerationError):
        generate_molecule()
    with pytest.raises(NotImplementedError):
        generate_molecule(molecule_name="N6H7")


def test_generate_molecule_from_name():
    # NRR intermediate
    mol = generate_molecule(molecule_name="NNH")
    assert mol["structure"].positions[0] == approx([7.345, 7.5, 6.545])
    # ORR intermediate
    mol = generate_molecule(molecule_name="OH")
    assert mol["structure"].positions[1][2] == approx(7.985)
    # atom-in-a-box
    mol = generate_molecule(molecule_name="Cl")
    assert len(mol["structure"]) == 1
    assert mol["structure"].positions[0][0] == approx(7.5)
    # ase g2 molecule
    mol = generate_molecule(molecule_name="C2H6SO")
    assert mol["structure"].positions[-1][0] == approx(6.2441475)


def test_generate_molecule_disk_io():
    _tmp_dir = tempfile.TemporaryDirectory().name
    mol = generate_molecule(
        molecule_name="N", write_to_disk=True, write_location=_tmp_dir
    )
    assert os.path.samefile(
        mol["traj_file_path"], os.path.join(_tmp_dir, "references", "N", "input.traj"),
    )


def test_place_adsorbate_defaults():
    pass


def test_place_adsorbate():
    pass


def test_get_adsorbate_height_estimate():
    # Tests height estimation based on covalent radii of nn
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    m = generate_molecule(molecule_name="O")["structure"]
    # Checks ontop corresponds to cov_rad1 + cov_rad2
    assert get_adsorbate_height_estimate(
        surface=surf, adsorbate=m, adsorption_site=(0.0, 0.0)
    ) == approx(2.02)
    # Checks hollow placement maintains distance of cov_rad1+cov_rad2
    surf = generate_surface_structures(["Cu"])["Cu"]["fcc111"]["structure"]
    m = generate_molecule(molecule_name="OH")["structure"]
    # Checks average estimated height based on nn is used
    assert get_adsorbate_height_estimate(
        surface=surf, adsorbate=m, adsorption_site=(3.82898322, 0.73688816)
    ) == approx(1.3222644717)


def test_get_adsorbate_slab_nn_list():
    # Tests generation of nn list for adsorbate placement
    # Checks number of nn identified
    surf = generate_surface_structures(["Cu"])["Cu"]["fcc111"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(
        surface=surf, adsorption_site=(3.82898322, 0.73688816)
    )
    assert len(nn_list) == 3
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(surface=surf, adsorption_site=(0.0, 0.0))
    assert len(nn_list) == 1
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(
        surface=surf, adsorption_site=(6.92964646, 4.15778787)
    )
    assert len(nn_list) == 4
    saa = generate_saa_structures(["Cu"], ["Pt"])["Cu"]["Pt"]["fcc111"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(
        surface=saa, adsorption_site=(7.01980257, 3.31599674)
    )
    # Checks correct nn identified
    assert nn_list[0][0] == "Pt"
    assert nn_list[1][0] == "Cu"
    assert nn_list[0][1][0] == approx(saa[27].x)
    assert nn_list[1][1][1] == approx(saa[28].y)


def test_get_adsorption_sites():
    pass


def test_generate_adsorbed_structures_data_structure():
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    ads = generate_adsorbed_structures(surface=surf, adsorbates=["NH", "OOH"])
    assert "NH" in ads
    assert "OOH" in ads


def test_generate_adsorbed_structures_manual_sites():
    # Test manual specification of adsorption sites
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["O"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)], "custom": [(0.5, 0.5)]},
    )
    assert ads["O"]["origin"]["0.0_0.0"]["structure"][-1].symbol == "O"
    assert ads["O"]["origin"]["0.0_0.0"]["structure"][-1].x == approx(0.0)
    assert ads["O"]["custom"]["0.5_0.5"]["structure"][-1].y == approx(0.5)


def test_generate_adsorbed_structures_ase_atoms_input():
    # Tests giving an Atoms object instead of a str as the adsorbate to be placed
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    m = molecule("CO")
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates={"CO": m, "H": "H"},
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        anchor_atom_indices={"CO": 1},
    )
    assert "CO" in ads
    assert len(ads["CO"]["origin"]["0.0_0.0"]["structure"]) == (len(surf) + len(m))
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-1].symbol == "C"
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-2].symbol == "O"


def test_generate_adsorbed_structures_mol_placement():
    # Tests default height
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
    )
    assert (
        ads["H"]["origin"]["0.0_0.0"]["structure"][-1].z
        - ads["H"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(1.67)
    # Tests manually specifying height
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["OH", "O"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        heights={"OH": 2.3},
    )
    assert (
        ads["OH"]["origin"]["0.0_0.0"]["structure"][-2].z
        - ads["OH"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(2.3)
    assert (
        ads["O"]["origin"]["0.0_0.0"]["structure"][-1].z
        - ads["O"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(2.02)
    # Tests manually specifying anchor_atom_indices
    m = molecule("CO")
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates={"CO": m},
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        anchor_atom_indices={"CO": 1},
    )
    assert (
        ads["CO"]["origin"]["0.0_0.0"]["structure"][-1].z
        - ads["CO"]["origin"]["0.0_0.0"]["structure"][27].z
    ) == approx(2.12)


def test_generate_adsorbed_structures_mol_rotation():
    # Tests applied rotations to adsorbates
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["NH3", "CO"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        heights={"CO": 1.5},
        rotations={"NH3": [[180.0, "x"], [90.0, "z"]], "CO": [[180.0, "y"]]},
    )
    # Check orientation of NH3
    assert ads["NH3"]["origin"]["0.0_0.0"]["structure"][-2].x == approx(-0.469865)
    assert ads["NH3"]["origin"]["0.0_0.0"]["structure"][-2].y == approx(0.813831)
    assert ads["NH3"]["origin"]["0.0_0.0"]["structure"][-2].z == approx(19.2479361)
    # Check orientation of CO
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-2].z == approx(18.28963917)
    assert ads["CO"]["origin"]["0.0_0.0"]["structure"][-1].z == approx(19.43997917)


def test_generate_adsorbed_structures_use_all_sites():
    # Test automated placement of adsorbate
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        use_all_sites=True,
        site_types=["ontop", "hollow"],
    )
    assert list(ads["H"].keys()) == ["ontop", "hollow"]
    assert list(ads["H"]["ontop"].keys()) == ["0.0_0.0"]
    # Test all sites automatically identified
    ads = generate_adsorbed_structures(
        surface=surf, adsorbates=["H"], use_all_sites=True
    )
    assert len(ads["H"]["ontop"]) == 1
    assert len(ads["H"]["hollow"]) == 2
    assert len(ads["H"]["bridge"]) == 1


def test_generate_adsorbed_structures_write_location():
    # Test user-specified write location
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["OH", "O"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0), (0.5, 0.5)], "custom": [(0.3, 0.3)]},
        write_to_disk=True,
        write_location="test_dir",
    )
    assert os.path.samefile(
        ads["OH"]["origin"]["0.0_0.0"]["traj_file_path"],
        "test_dir/adsorbates/OH/origin/0.0_0.0/input.traj",
    )
    assert os.path.samefile(
        ads["O"]["origin"]["0.5_0.5"]["traj_file_path"],
        "test_dir/adsorbates/O/origin/0.5_0.5/input.traj",
    )
    assert os.path.samefile(
        ads["O"]["custom"]["0.3_0.3"]["traj_file_path"],
        "test_dir/adsorbates/O/custom/0.3_0.3/input.traj",
    )
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["OH"],
        site_types=["bridge"],
        write_to_disk=True,
        write_location="test_dir",
    )
    assert os.path.samefile(
        ads["OH"]["bridge"]["7.623_6.001"]["traj_file_path"],
        "test_dir/adsorbates/OH/bridge/7.623_6.001/input.traj",
    )
    shutil.rmtree("test_dir")


def test_generate_adsorbed_structures_dirs_exist_ok():
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        write_to_disk=True,
    )
    with raises(FileExistsError):
        ads = generate_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            use_all_sites=False,
            adsorption_sites={"origin": [(0.0, 0.0)]},
            write_to_disk=True,
        )
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        write_to_disk=True,
        dirs_exist_ok=True,
    )
    assert os.path.samefile(
        ads["H"]["origin"]["0.0_0.0"]["traj_file_path"],
        "adsorbates/H/origin/0.0_0.0/input.traj",
    )
    shutil.rmtree("adsorbates")
