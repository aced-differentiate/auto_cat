"""Unit tests for the `autocat.adsorption` module."""

import os
import tempfile

from pytest import approx
from pytest import raises

import numpy as np

from ase.build import molecule
from ase import Atoms

from autocat.saa import generate_saa_structures
from autocat.surface import generate_surface_structures
from autocat.adsorption import generate_molecule
from autocat.adsorption import generate_adsorbed_structures
from autocat.adsorption import place_adsorbate
from autocat.adsorption import get_adsorbate_height_estimate
from autocat.adsorption import get_adsorbate_slab_nn_list
from autocat.adsorption import get_adsorption_sites
from autocat.adsorption import AutocatAdsorptionGenerationError
from autocat.adsorption import adsorption_sites_to_possible_ads_site_list
from autocat.adsorption import enumerate_adsorbed_site_list
from autocat.adsorption import place_multiple_adsorbates
from autocat.adsorption import generate_high_coverage_adsorbed_structures


def test_generate_molecule_from_name_error():
    with raises(AutocatAdsorptionGenerationError):
        generate_molecule()
    with raises(NotImplementedError):
        generate_molecule(molecule_name="N6H7")


def test_generate_molecule_from_name():
    # NRR intermediate
    mol = generate_molecule(molecule_name="NNH")
    assert mol["structure"].positions[0] == approx([7.345, 7.5, 6.545])
    # ORR intermediate
    mol = generate_molecule(molecule_name="OH")
    assert mol["structure"].positions[1][2] == approx(7.845)
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
    traj_file_path = os.path.join(_tmp_dir, "references", "N", "input.traj")
    assert os.path.samefile(mol["traj_file_path"], traj_file_path)


def test_adsorption_sites_to_possible_ads_site_list():
    # Test converting adsorption sites to appropriately formatted list

    # test missing adsorbates
    with raises(AutocatAdsorptionGenerationError):
        adsorption_sites_to_possible_ads_site_list(
            adsorption_sites=[(0.25, 0.25), (0.0, 0.0)]
        )

    # test missing adsorption sites
    with raises(AutocatAdsorptionGenerationError):
        adsorption_sites_to_possible_ads_site_list(adsorbates=["N"])

    # test giving adsorption_sites as a list
    poss_ads, sites = adsorption_sites_to_possible_ads_site_list(
        adsorption_sites=[(0.3, 0.3), (0.9, 1.4)], adsorbates=["OH", "OOH", "C"]
    )
    assert sites == [(0.3, 0.3), (0.9, 1.4)]
    assert poss_ads == [["OH", "OOH", "C"], ["OH", "OOH", "C"]]

    # test pruning adsorption_sites as a list for repeated sites
    poss_ads, sites = adsorption_sites_to_possible_ads_site_list(
        adsorption_sites=[(0.3, 0.3), (0.9, 1.4), (0.3, 0.3)],
        adsorbates=["OH", "OOH", "C"],
    )
    assert sites == [(0.3, 0.3), (0.9, 1.4)]
    assert poss_ads == [["OH", "OOH", "C"], ["OH", "OOH", "C"]]

    # test giving adsorbates as a dict
    poss_ads, sites = adsorption_sites_to_possible_ads_site_list(
        adsorption_sites=[(0.3, 0.3), (0.9, 1.4)],
        adsorbates={"OH_1": "OH", "OH_2": "OH", "C*": "C"},
    )
    assert sites == [(0.3, 0.3), (0.9, 1.4)]
    assert poss_ads == [["OH_1", "OH_2", "C*"], ["OH_1", "OH_2", "C*"]]

    # test giving adsorption sites as a dict
    poss_ads, sites = adsorption_sites_to_possible_ads_site_list(
        adsorption_sites={"OH": [(0.0, 0.0)], "O": [(0.0, 0.0), (0.5, 0.5)]}
    )
    assert sites == [(0.0, 0.0), (0.5, 0.5)]
    assert poss_ads == [["OH", "O"], ["O"]]

    # test giving both arguments as dicts
    poss_ads, sites = adsorption_sites_to_possible_ads_site_list(
        adsorption_sites={
            "OH_1": [(0.0, 0.0)],
            "OH_2": [(0.0, 0.0), (0.5, 0.5)],
            "C*": [(0.5, 0.5)],
        },
        adsorbates={"OH_1": "OH", "OH_2": "OH", "C*": "C"},
    )
    assert sites == [(0.0, 0.0), (0.5, 0.5)]
    assert poss_ads == [["OH_1", "OH_2"], ["OH_2", "C*"]]


def test_enumerate_adsorbed_site_list_invalid_inputs():
    # no adsorbates
    with raises(AutocatAdsorptionGenerationError):
        enumerate_adsorbed_site_list(
            adsorption_sites={"OH": [[0.5, 0.5], [0.6, 0.1]]},
            adsorbate_coverage={"OH": 2},
        )
    # no adsorbate coverage
    with raises(AutocatAdsorptionGenerationError):
        enumerate_adsorbed_site_list(
            adsorption_sites={"OH": [[0.5, 0.5], [0.6, 0.1]]}, adsorbates=["OH"]
        )
    # adsorbate coverage not given for an adsorbate
    with raises(AutocatAdsorptionGenerationError):
        enumerate_adsorbed_site_list(
            adsorption_sites={"OH": [[0.5, 0.5], [0.6, 0.1]]},
            adsorbate_coverage={"OH": 2},
            adsorbates=["OH", "C"],
        )
    with raises(AutocatAdsorptionGenerationError):
        enumerate_adsorbed_site_list(
            adsorption_sites={"OH_1": [[0.5, 0.5], [0.6, 0.1]]},
            adsorbate_coverage={"OH_1": 2},
            adsorbates={"OH_1": "OH", "C*": "C"},
        )


def test_enumerate_adsorbed_site_list():
    # cov given as num of adsorbates
    ads_site_list, sites = enumerate_adsorbed_site_list(
        adsorbates=["CO", "OH"],
        adsorbate_coverage={"CO": 2, "OH": 1},
        adsorption_sites=[(0.0, 0.0), (0.2, 0.1), (0.0, 0.4)],
    )
    assert sites == [(0.0, 0.0), (0.2, 0.1), (0.0, 0.4)]
    assert ads_site_list == [["CO", "CO", "OH"], ["CO", "OH", "CO"], ["OH", "CO", "CO"]]

    # cov given as fractions
    ads_site_list, sites = enumerate_adsorbed_site_list(
        adsorbates=["CO", "OH"],
        adsorbate_coverage={"CO": 0.7, "OH": 0.4},
        adsorption_sites=[(0.0, 0.0), (0.2, 0.1), (0.0, 0.4)],
    )
    assert sites == [(0.0, 0.0), (0.2, 0.1), (0.0, 0.4)]
    assert ads_site_list == [["CO", "CO", "OH"], ["CO", "OH", "CO"], ["OH", "CO", "CO"]]

    # ensure not going over max cov num
    ads_site_list, sites = enumerate_adsorbed_site_list(
        adsorbates=["CO", "OH"],
        adsorbate_coverage={"CO": 2, "OH": 1},
        adsorption_sites=[(0.0, 0.0), (0.2, 0.1)],
    )
    assert sites == [(0.0, 0.0), (0.2, 0.1)]
    assert ads_site_list == [["CO", "CO"], ["CO", "OH"], ["OH", "CO"]]

    # ensure not going over max cov fraction
    ads_site_list, sites = enumerate_adsorbed_site_list(
        adsorbates=["CO", "OH"],
        adsorbate_coverage={"CO": 1, "OH": 0.5},
        adsorption_sites=[(0.0, 0.0), (0.2, 0.1)],
    )
    assert sites == [(0.0, 0.0), (0.2, 0.1)]
    assert ads_site_list == [["CO", "CO"], ["CO", "OH"], ["OH", "CO"]]

    # catches too restrictive cov fraction
    with raises(AutocatAdsorptionGenerationError):
        ads_site_list, sites = enumerate_adsorbed_site_list(
            adsorbates=["CO"],
            adsorbate_coverage={"CO": 0.4},
            adsorption_sites=[(0.0, 0.0), (0.2, 0.1)],
        )


def test_place_multi_adsorbates_invalid_inputs():
    surf = generate_surface_structures(species_list=["Fe"])["Fe"]["bcc100"]["structure"]
    # no surface
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(surface=None)
    # no adsorbates
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(surface=surf, adsorbates=None)
    # no adsorbate sites list
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf, adsorbates=["OH", "O"], adsorption_sites_list=None
        )
    # no adsorbates at each site list
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf,
            adsorbates=["OH", "O"],
            adsorption_sites_list=[(0.0, 0.0), (0.3, 0.4)],
            adsorbates_at_each_site=None,
        )
    # wrong adsorbate type
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf,
            adsorbates={"OH", "O"},
            adsorption_sites_list=[(0.0, 0.0), (0.3, 0.4)],
            adsorbates_at_each_site=["OH", "O"],
        )
    # adsorbates at each site given in incorrect fmt
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf,
            adsorbates=["OH", "O"],
            adsorption_sites_list=[(0.0, 0.0), (0.3, 0.4)],
            adsorbates_at_each_site={"OH": [(0.0, 0.0)], "O": [(0.3, 0.4)]},
        )
    # some adsorbates in adsorbates at each site given as Atoms
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf,
            adsorbates=["OH", "O"],
            adsorption_sites_list=[(0.0, 0.0), (0.3, 0.4)],
            adsorbates_at_each_site=["OH", Atoms("O")],
        )
    # sites and list of adsorbates at each site do not match
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf,
            adsorbates=["OH", "O"],
            adsorption_sites_list=[(0.0, 0.0), (0.3, 0.4)],
            adsorbates_at_each_site=["OH"],
        )
    # wrong fmt of element in adsorbates
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf,
            adsorbates={"OH": "OH", "C*": ["C"]},
            adsorption_sites_list=[(0.0, 0.0), (0.3, 0.4)],
            adsorbates_at_each_site=["OH", "C*"],
        )
    # multiple adsorbates placed at same site
    with raises(AutocatAdsorptionGenerationError):
        place_multiple_adsorbates(
            surface=surf,
            adsorbates=["OH", "N"],
            adsorption_sites_list=[(0.0, 0.0), (0.3, 0.4), (0.0, 0.0)],
            adsorbates_at_each_site=["OH", "OH", "N"],
        )


def test_place_multi_adsorbates_placement():
    # test that adsorbates are placed in the right locations
    surf = generate_surface_structures(species_list=["Fe"])["Fe"]["bcc100"]["structure"]
    # adsorbates given as list
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates=["O", "H"],
        adsorbates_at_each_site=["O", "H"],
        adsorption_sites_list=[(0.0, 0.0), (0.5, 0.5)],
    )
    assert "O" in ads_multi.get_chemical_symbols()
    assert "H" in ads_multi.get_chemical_symbols()
    assert ads_multi[-2].symbol == "O"
    assert np.isclose(ads_multi[-2].x, 0.0)
    assert np.isclose(ads_multi[-2].y, 0.0)
    assert ads_multi[-1].symbol == "H"
    assert np.isclose(ads_multi[-1].x, 0.5)
    assert np.isclose(ads_multi[-1].y, 0.5)
    # adsorbates given as dict
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates={"O*": "O", "H*": Atoms("H")},
        adsorbates_at_each_site=["O*", "H*"],
        adsorption_sites_list=[(0.1, 0.2), (0.7, 0.0)],
    )
    assert "O" in ads_multi.get_chemical_symbols()
    assert "H" in ads_multi.get_chemical_symbols()
    assert ads_multi[-2].symbol == "O"
    assert np.isclose(ads_multi[-2].x, 0.1)
    assert np.isclose(ads_multi[-2].y, 0.2)
    assert ads_multi[-1].symbol == "H"
    assert np.isclose(ads_multi[-1].x, 0.7)
    assert np.isclose(ads_multi[-1].y, 0.0)


def test_place_multi_ads_height_and_anchor_idx():
    surf = generate_surface_structures(species_list=["Fe"])["Fe"]["bcc100"]["structure"]
    # height given as float
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates=["O", "C"],
        adsorbates_at_each_site=["O", "C"],
        adsorption_sites_list=[(0.0, 0.0), (2.87, 2.87)],
        heights=0.5,
    )
    assert np.isclose(ads_multi[-1].z, 14.805)
    assert np.isclose(ads_multi[-2].z, 14.805)
    # height given as dict
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates=["O", "C"],
        adsorbates_at_each_site=["O", "C"],
        adsorption_sites_list=[(0.0, 0.0), (2.87, 2.87)],
        heights={"O": 1.0, "C": 1.5},
    )
    assert np.isclose(ads_multi[-1].z, 15.805)
    assert np.isclose(ads_multi[-2].z, 15.305)
    # anchor idx specified
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates=["OH", "CO"],
        adsorbates_at_each_site=["OH", "CO"],
        adsorption_sites_list=[(0.0, 0.0), (2.87, 2.87)],
        heights={"OH": 1.0, "CO": 2.5},
        anchor_atom_indices={"CO": 1},
    )
    assert np.isclose(ads_multi[-4].z, 15.305)
    assert np.isclose(ads_multi[-1].z, 16.805)
    # anchor idx given for all adsorbates
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates=["OH", "CO"],
        adsorbates_at_each_site=["OH", "CO"],
        adsorption_sites_list=[(0.0, 0.0), (2.87, 2.87)],
        heights=1.5,
        anchor_atom_indices=1,
    )
    assert np.isclose(ads_multi[-3].z, 15.805)
    assert np.isclose(ads_multi[-1].z, 15.805)


def test_place_multi_ads_rotations():
    surf = generate_surface_structures(species_list=["Fe"])["Fe"]["bcc100"]["structure"]
    # same rotation applied to all adsorbate types
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates=["OH", "NH"],
        adsorbates_at_each_site=["OH", "NH"],
        adsorption_sites_list=[(0.0, 0.0), (2.87, 2.87)],
        rotations=[(45.0, "x")],
    )
    assert np.isclose(ads_multi[-3].y, -0.4879036790187179)
    assert np.isclose(ads_multi[-3].z, 16.772903679018718)
    assert np.isclose(ads_multi[-1].y, 2.3679541853575516)
    assert np.isclose(ads_multi[-1].x, 3.58)
    # rotations given by dict
    ads_multi = place_multiple_adsorbates(
        surface=surf,
        adsorbates=["OH", "NH"],
        adsorbates_at_each_site=["OH", "NH"],
        adsorption_sites_list=[(0.0, 0.0), (2.87, 2.87)],
        rotations={"OH": [(45.0, "x")], "NH": [(90.0, "z")]},
    )
    assert np.isclose(ads_multi[-3].y, -0.4879036790187179)
    assert np.isclose(ads_multi[-3].z, 16.772903679018718)
    assert np.isclose(ads_multi[-1].z, 17.045)
    assert np.isclose(ads_multi[-1].x, 2.87)


def test_generate_high_cov_invalid_inputs():
    surf = generate_surface_structures(species_list=["Fe"])["Fe"]["bcc100"]["structure"]
    # no surface
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures()
    # incorrect surface type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(surface="Fe")
    # no adsorbates input
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(surface=surf)
    # incorrect adsorbates type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(surface=surf, adsorbates=surf)
    # no adsorbate coverage
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(surface=surf, adsorbates=["H"])
    # incorrect adsorbate coverage type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf, adsorbates=["H"], adsorbate_coverage=0.5
        )
    # incorrect rotations type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf, adsorbates=["H"], adsorbate_coverage={"H": 9}, rotations="x"
        )
    # incorrect adsorption_sites type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            adsorption_sites=(0, 0),
        )
    # incorrect use_all_sites type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            use_all_sites=[True, True],
        )
    # incorrect site_type input
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            site_types="anywhere",
        )
    # incorrect site_type type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            site_types=5,
        )
    # incorrect site_type value in dict
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            site_types={"H": [0.5, "ontop"]},
        )
    # incorrect site_type dict fmt
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            site_types={"H": 0.5},
        )
    # incorrect site_types value in list
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            site_types=["ontop", "anywhere"],
        )
    # incorrect heights type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            heights=[1.5],
        )
    # incorrect anchor_atom_indices type
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            adsorbate_coverage={"H": np.inf},
            anchor_atom_indices=[0, 1],
        )
    # adsorbate not in adsorbate_coverage
    with raises(AutocatAdsorptionGenerationError):
        generate_high_coverage_adsorbed_structures(
            surface=surf, adsorbates=["H", "X"], adsorbate_coverage={"H": np.inf},
        )


def test_generate_high_cov_valid_inputs():
    surf = generate_surface_structures(species_list=["Fe"], supercell_dim=(2, 2, 3))[
        "Fe"
    ]["bcc100"]["structure"]
    # rotations as a list
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["OH"],
        adsorption_sites={"OH": [(0.0, 0.0)]},
        adsorbate_coverage={"OH": 1},
        use_all_sites=False,
        site_types=["ontop"],
        rotations=[(45.0, "x")],
    )
    assert np.isclose(multi_ads_dict[0]["structure"][-1].y, -0.4879036790187179)
    # site types as a str
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        adsorbate_coverage={"H": np.inf},
        use_all_sites=True,
        site_types="ontop",
    )
    assert len(multi_ads_dict[0]["structure"]) == len(surf) + 4
    # heights as float
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H", "C"],
        adsorption_sites={"H": [(0.0, 0.0)], "C": [(2.87, 2.87)]},
        adsorbate_coverage={"H": 1, "C": 1},
        heights=1.5,
    )
    assert np.isclose(multi_ads_dict[0]["structure"][-2].z, 14.37)
    assert np.isclose(multi_ads_dict[0]["structure"][-1].z, 14.37)
    # anchor atom idx as int
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["OH", "CO"],
        adsorption_sites={"OH": [(0.0, 0.0)], "CO": [(2.87, 2.87)]},
        adsorbate_coverage={"OH": 1, "CO": 1},
        heights=1.5,
        anchor_atom_indices=1,
    )
    assert np.isclose(multi_ads_dict[0]["structure"][-3].z, 14.37)
    assert np.isclose(multi_ads_dict[0]["structure"][-1].z, 14.37)


def test_generate_high_cov_write_location():
    surf = generate_surface_structures(species_list=["Fe"], supercell_dim=(2, 2, 3))[
        "Fe"
    ]["bcc100"]["structure"]
    _tmp_dir = tempfile.TemporaryDirectory().name
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H", "X"],
        adsorbate_coverage={"H": 11, "X": 1},
        use_all_sites=True,
        site_types=["bridge"],
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    traj_file_path = os.path.join(_tmp_dir, "multiple_adsorbates", "0", "input.traj")
    assert os.path.samefile(multi_ads_dict[0]["traj_file_path"], traj_file_path)
    traj_file_path = os.path.join(_tmp_dir, "multiple_adsorbates", "1", "input.traj")
    assert os.path.samefile(multi_ads_dict[1]["traj_file_path"], traj_file_path)


def test_generate_high_cov_dirs_exist_ok():
    surf = generate_surface_structures(species_list=["Fe"], supercell_dim=(2, 2, 3))[
        "Fe"
    ]["bcc100"]["structure"]
    _tmp_dir = tempfile.TemporaryDirectory().name
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H", "X"],
        adsorbate_coverage={"H": 11, "X": 1},
        use_all_sites=True,
        site_types=["bridge"],
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    with raises(FileExistsError):
        multi_ads_dict = generate_high_coverage_adsorbed_structures(
            surface=surf,
            adsorbates=["H", "X"],
            adsorbate_coverage={"H": 11, "X": 1},
            use_all_sites=True,
            site_types=["bridge"],
            write_to_disk=True,
            write_location=_tmp_dir,
        )
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H", "X"],
        adsorbate_coverage={"H": 11, "X": 1},
        use_all_sites=True,
        site_types=["bridge"],
        write_to_disk=True,
        write_location=_tmp_dir,
        dirs_exist_ok=True,
    )
    traj_file_path = os.path.join(_tmp_dir, "multiple_adsorbates", "0", "input.traj")
    assert os.path.samefile(multi_ads_dict[0]["traj_file_path"], traj_file_path)


def test_generate_high_cov_use_all_sites():
    surf = generate_surface_structures(species_list=["Fe"], supercell_dim=(2, 2, 3))[
        "Fe"
    ]["bcc100"]["structure"]
    # test filling all top sites
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        adsorbate_coverage={"H": np.inf},
        use_all_sites=True,
        site_types=["ontop"],
    )
    assert len(multi_ads_dict) == 1
    struct = multi_ads_dict[0]["structure"]
    assert list(struct.symbols).count("H") == 4
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H", "X"],
        adsorbate_coverage={"H": 11, "X": 1},
        use_all_sites=True,
        site_types=["bridge"],
    )
    # has only 2 unique bridge sites
    # so by symm should only be 1 struct for each
    assert len(multi_ads_dict) == 2
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H", "X"],
        adsorbate_coverage={"H": 0.5, "X": 0.5},
        use_all_sites=True,
        site_types=["ontop"],
    )
    # has 1 unique top sites (4 total)
    # so by symm should only be 2 struct
    # (diagonal and straight)
    assert len(multi_ads_dict) == 2

    # test with variable vacancies
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H", "X"],
        adsorbate_coverage={"H": 0.5, "X": np.inf},
        use_all_sites=True,
        site_types=["ontop"],
    )
    # has 1 unique top sites (4 total)
    # so by symm should only be 3 struct
    # (2 with 2 adsorbates and 1 with 1 adsorbate)
    assert len(multi_ads_dict) == 3


def test_generate_high_cov_rm_X():
    # test filtering out X to get vacancies
    surf = generate_surface_structures(species_list=["Fe"], supercell_dim=(2, 2, 3))[
        "Fe"
    ]["bcc100"]["structure"]
    # test filling all top sites
    multi_ads_dict = generate_high_coverage_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        adsorbate_coverage={"H": 3, "X": np.inf},
        use_all_sites=True,
        site_types=["ontop"],
    )
    struct = multi_ads_dict[0]["structure"]
    assert "X" not in list(struct.symbols)


def test_generate_adsorbed_structures_invalid_inputs():
    surf = generate_surface_structures(species_list=["Fe"])["Fe"]["bcc100"]["structure"]
    # no surface input
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures()
    # incorrect surface type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(surface="Fe")
    # no adsorbates input
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(surface=surf)
    # incorrect adsorbates type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(surface=surf, adsorbates=surf)
    # incorrect rotations type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(surface=surf, adsorbates=["H"], rotations="x")
    # incorrect adsorption_sites type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(
            surface=surf, adsorbates=["H"], adsorption_sites=[(0, 0)]
        )
    # incorrect use_all_sites type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(
            surface=surf, adsorbates=["H"], use_all_sites=[True, True]
        )
    # incorrect site_types type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(
            surface=surf, adsorbates=["H"], site_types="bridge"
        )
    # incorrect heights type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(surface=surf, adsorbates=["H"], heights=[1.5])
    # incorrect anchor_atom_indices type
    with raises(AutocatAdsorptionGenerationError):
        generate_adsorbed_structures(
            surface=surf, adsorbates=["H"], anchor_atom_indices=[0, 1]
        )


def test_generate_adsorbed_structures_valid_input_types():
    surf = generate_surface_structures(species_list=["Fe"])["Fe"]["bcc100"]["structure"]
    # single rotations list input
    ads = generate_adsorbed_structures(
        surface=surf, adsorbates=["CO2"], rotations=[(90, "x")]
    )
    assert ads["CO2"]["origin"]["0_0"]["structure"].positions[-1][2] == approx(16.385)
    # single heights value input
    ads = generate_adsorbed_structures(surface=surf, adsorbates=["CO2"], heights=3.5)
    assert ads["CO2"]["origin"]["0_0"]["structure"].positions[-1][2] == approx(
        16.626342
    )
    # single anchor_atom_index input
    ads = generate_adsorbed_structures(
        surface=surf, adsorbates=["CO2"], anchor_atom_indices=1
    )
    assert ads["CO2"]["origin"]["0_0"]["structure"].positions[-1][2] == approx(
        13.927684
    )


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
    _tmp_dir = tempfile.TemporaryDirectory().name
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["OH", "O"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0), (0.5, 0.5)], "custom": [(0.3, 0.3)]},
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    traj_file_path = os.path.join(
        _tmp_dir, "adsorbates", "OH", "origin", "0.0_0.0", "input.traj"
    )
    assert os.path.samefile(
        ads["OH"]["origin"]["0.0_0.0"]["traj_file_path"], traj_file_path
    )
    traj_file_path = os.path.join(
        _tmp_dir, "adsorbates", "O", "origin", "0.5_0.5", "input.traj"
    )
    assert os.path.samefile(
        ads["O"]["origin"]["0.5_0.5"]["traj_file_path"], traj_file_path
    )
    traj_file_path = os.path.join(
        _tmp_dir, "adsorbates", "O", "custom", "0.3_0.3", "input.traj"
    )
    assert os.path.samefile(
        ads["O"]["custom"]["0.3_0.3"]["traj_file_path"], traj_file_path
    )
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["OH"],
        site_types=["bridge"],
        use_all_sites=True,
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    loc = list(ads["OH"]["bridge"].keys())[0]
    traj_file_path = os.path.join(
        _tmp_dir, "adsorbates", "OH", "bridge", loc, "input.traj"
    )
    assert os.path.samefile(ads["OH"]["bridge"][loc]["traj_file_path"], traj_file_path)


def test_generate_adsorbed_structures_dirs_exist_ok():
    _tmp_dir = tempfile.TemporaryDirectory().name
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    with raises(FileExistsError):
        ads = generate_adsorbed_structures(
            surface=surf,
            adsorbates=["H"],
            use_all_sites=False,
            adsorption_sites={"origin": [(0.0, 0.0)]},
            write_to_disk=True,
            write_location=_tmp_dir,
        )
    ads = generate_adsorbed_structures(
        surface=surf,
        adsorbates=["H"],
        use_all_sites=False,
        adsorption_sites={"origin": [(0.0, 0.0)]},
        write_to_disk=True,
        write_location=_tmp_dir,
        dirs_exist_ok=True,
    )
    traj_file_path = os.path.join(
        _tmp_dir, "adsorbates", "H", "origin", "0.0_0.0", "input.traj"
    )
    assert os.path.samefile(
        ads["H"]["origin"]["0.0_0.0"]["traj_file_path"], traj_file_path
    )


def test_place_adsorbate_defaults():
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    mol = molecule("NH3")
    ads = place_adsorbate(surface=surf, adsorbate=mol)
    # check that the first atom in the molecule (N) is at the origin (default
    # adsorption_site)
    assert ads.positions[-4][:2] == approx([0, 0])
    # check that the molecule is placed on the surface without rotation.
    # By default NH3 is in the following configuration:  N
    #                                                  / | \
    #                                                 H  H  H
    #                                                 surface
    assert ads.positions[-1][2] == approx(ads.positions[-2][2])
    assert ads.positions[-1][0] == approx(-1 * ads.positions[-2][0])


def test_get_adsorbate_height_estimate_defaults():
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    mol = molecule("NH3")
    default_h = get_adsorbate_height_estimate(surface=surf, adsorbate=mol)
    origin_h = get_adsorbate_height_estimate(
        surface=surf, adsorbate=mol, adsorption_site=(0, 0)
    )
    assert default_h == approx(origin_h)


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


def test_get_adsorbate_slab_nn_list_defaults():
    surf = generate_surface_structures(["Cu"])["Cu"]["fcc111"]["structure"]
    default_nn_list = get_adsorbate_slab_nn_list(surface=surf)
    origin_nn_list = get_adsorbate_slab_nn_list(surface=surf, adsorption_site=(0, 0))
    # check that the default xy coordinates of the nn Cu atom is at the origin
    assert default_nn_list[1][0][0] == origin_nn_list[1][0][0]
    assert default_nn_list[1][0][1] == origin_nn_list[1][0][1]


def test_get_adsorbate_slab_nn_list():
    # Tests generation of nn list for adsorbate placement
    # Checks number of nn identified
    surf = generate_surface_structures(["Cu"])["Cu"]["fcc111"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(
        surface=surf, adsorption_site=(3.82898322, 0.73688816)
    )
    assert len(nn_list[0]) == 3
    # Checks that 3 positions given (1 for each nn)
    assert len(nn_list[1]) == 3
    assert all([nn_list[1][i].shape == (3,) for i in range(len(nn_list[0]))])

    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(surface=surf, adsorption_site=(0.0, 0.0))
    # Check that 1 nn identified
    assert len(nn_list[0]) == 1
    # Checks that 1 position given (1 for each nn)
    assert len(nn_list[1]) == 1
    assert all([nn_list[1][i].shape == (3,) for i in range(len(nn_list[0]))])

    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(
        surface=surf, adsorption_site=(6.92964646, 4.15778787),
    )
    # Checks that 4 nn identified
    assert len(nn_list[0]) == 4
    # Checks that 4 position given (1 for each nn)
    assert len(nn_list[1]) == 4
    assert all([nn_list[1][i].shape == (3,) for i in range(len(nn_list[0]))])

    saa = generate_saa_structures(["Cu"], ["Pt"])["Cu"]["Pt"]["fcc111"]["structure"]
    nn_list = get_adsorbate_slab_nn_list(
        surface=saa, adsorption_site=(7.01980257, 3.31599674)
    )
    # Checks correct nn identified
    assert nn_list[0][0] == "Pt"
    assert nn_list[0][1] == "Cu"
    assert nn_list[1][0][0] == approx(saa[27].x)
    assert nn_list[1][1][1] == approx(saa[28].y)


def test_get_adsorption_sites_defaults():
    surf = generate_surface_structures(["Cu"])["Cu"]["fcc111"]["structure"]
    ads_sites = get_adsorption_sites(surface=surf)
    default_site_types = ["bridge", "hollow", "ontop"]
    assert not set(ads_sites.keys()).difference(set(default_site_types))
