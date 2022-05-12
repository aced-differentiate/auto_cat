"""Unit tests for the `autocat.perturbations` module"""

import os
import pytest
import numpy as np

import tempfile

from ase import Atoms
from ase.build import molecule

from autocat.surface import generate_surface_structures
from autocat.adsorption import place_adsorbate
from autocat.perturbations import AutocatPerturbationError, perturb_structure
from autocat.perturbations import generate_perturbed_dataset


def test_perturb_structure_directions():
    # Tests fixing directions of perturbations
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    mol = molecule("H")
    base_struct = place_adsorbate(surface=sub, adsorbate=mol)
    # free in all directions
    p_struct = perturb_structure(base_struct,)
    assert (p_struct["structure"].positions[-1] != base_struct.positions[-1]).all()
    # free in z
    base_struct[-1].tag = -1
    p_struct = perturb_structure(base_struct,)
    assert p_struct["structure"].positions[-1][0] == base_struct.positions[-1][0]
    assert p_struct["structure"].positions[-1][1] == base_struct.positions[-1][1]
    assert p_struct["structure"].positions[-1][-1] != base_struct.positions[-1][-1]


def test_perturb_structure_sign_constraint():
    # Tests fixing sign of perturbation
    sub = generate_surface_structures(["Cu"], facets={"Cu": ["111"]})["Cu"]["fcc111"][
        "structure"
    ]
    mol = molecule("O")
    base_struct = place_adsorbate(surface=sub, adsorbate=mol)
    # free in +z
    base_struct[-1].tag = -1
    p_struct = perturb_structure(base_struct, direction_sign_constraint=1)
    assert p_struct["structure"].positions[-1][0] == base_struct.positions[-1][0]
    assert p_struct["structure"].positions[-1][-1] - base_struct.positions[-1][-1] > 0.0
    # free in -x
    base_struct[-1].tag = -3
    p_struct = perturb_structure(base_struct, direction_sign_constraint=-1)
    assert p_struct["structure"].positions[-1][0] - base_struct.positions[-1][0] < 0.0
    # free in +xy
    base_struct[-1].tag = -2
    p_struct = perturb_structure(base_struct, direction_sign_constraint=1)
    assert p_struct["structure"].positions[-1][0] - base_struct.positions[-1][0] > 0.0
    assert p_struct["structure"].positions[-1][1] - base_struct.positions[-1][1] > 0.0
    with pytest.raises(AutocatPerturbationError):
        p_struct = perturb_structure(base_struct, direction_sign_constraint=2)


def test_perturb_structure_matrix():
    # Tests matrix matches perturbed structure
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct = place_adsorbate(surface=sub, adsorbate=mol)
    p_struct = perturb_structure(base_struct)
    o_pert = base_struct.positions[-2] + p_struct["perturbation_matrix"][-2]
    assert np.allclose(p_struct["structure"].positions[-2], o_pert)
    h_pert = base_struct.positions[-1] + p_struct["perturbation_matrix"][-1]
    assert np.allclose(p_struct["structure"].positions[-1], h_pert)


def test_generate_perturbed_dataset_num_of_perturbations():
    # Tests number of perturbations generated
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct = place_adsorbate(surface=sub, adsorbate=mol)
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=15,)
    assert len(p_set["HOPt36_0"].keys()) == 15


def test_generate_perturbed_dataset_multiple_base_structures():
    # Tests giving multiple base_structures
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct1 = place_adsorbate(surface=sub1, adsorbate=mol)
    mol = molecule("NH")
    base_struct2 = place_adsorbate(surface=sub2, adsorbate=mol)
    base_struct1[-2].tag = 1
    base_struct1[-1].tag = -1
    base_struct2[-1].tag = 1
    p_set = generate_perturbed_dataset([base_struct1, base_struct2],)
    # Check all base structures perturbed
    assert "HCu36N_1" in p_set
    assert "HOPt36_0" in p_set
    # Check correct direction constraints applied to each base_structure
    assert np.isclose(p_set["HOPt36_0"]["6"]["perturbation_matrix"][-1][0], 0.0)
    assert np.isclose(p_set["HOPt36_0"]["6"]["perturbation_matrix"][-1][1], 0.0)
    assert not np.isclose(p_set["HOPt36_0"]["6"]["perturbation_matrix"][-1][-1], 0.0)
    assert not np.isclose(p_set["HCu36N_1"]["1"]["perturbation_matrix"][-1][0], 0.0)


def test_generate_perturbed_dataset_write_location():
    # Tests write location
    _tmp_dir = tempfile.TemporaryDirectory().name
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct = place_adsorbate(surface=sub, adsorbate=mol)
    p_set = generate_perturbed_dataset(
        [base_struct], write_to_disk=True, write_location=_tmp_dir,
    )
    assert os.path.samefile(
        p_set["HOPt36_0"]["0"]["traj_file_path"],
        os.path.join(_tmp_dir, "HOPt36_0/0/perturbed_structure.traj"),
    )
    assert os.path.samefile(
        p_set["HOPt36_0"]["0"]["pert_mat_file_path"],
        os.path.join(_tmp_dir, "HOPt36_0/0/perturbation_matrix.json"),
    )
    assert os.path.samefile(
        p_set["correction_matrix_path"],
        os.path.join(_tmp_dir, "correction_matrix.json"),
    )


def test_generate_perturbed_dataset_correction_matrix():
    # Check that matrices properly collected for 1 base_structure
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct = place_adsorbate(surface=sub, adsorbate=mol)
    p_set = generate_perturbed_dataset([base_struct], num_of_perturbations=5,)
    manual_collect = np.zeros((5, 6))
    for idx, struct in enumerate(p_set["HOPt36_0"]):
        flat = p_set["HOPt36_0"][struct]["perturbation_matrix"].flatten()
        manual_collect[idx, : len(flat)] = flat
    assert np.allclose(p_set["correction_matrix"], manual_collect)
    assert p_set["correction_matrix"].shape == (5, 6)
    # Check when given specific maximum_structure_size
    p_set = generate_perturbed_dataset(
        [base_struct], num_of_perturbations=5, maximum_adsorbate_size=15,
    )
    assert p_set["correction_matrix"].shape == (5, 45)


def test_generate_perturbed_dataset_correction_matrix_multiple():
    # check that matrices properly collected for 2 base structures of == size
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct1 = place_adsorbate(surface=sub1, adsorbate=mol)
    base_struct2 = place_adsorbate(surface=sub2, adsorbate=mol)
    base_struct2[-1].tag = 1
    p_set = generate_perturbed_dataset(
        [base_struct1, base_struct2], num_of_perturbations=5,
    )
    manual_collect = np.zeros((10, 6))
    counter = 0
    for base in ["HOPt36_0", "HCu36O_1"]:
        for struct in p_set[base]:
            flat = p_set[base][struct]["perturbation_matrix"].flatten()
            manual_collect[counter, : len(flat)] = flat
            counter += 1
    assert np.allclose(p_set["correction_matrix"], manual_collect)
    assert p_set["correction_matrix"].shape == (10, 6)

    # check that matrices properly collected for 2 base structures of != size
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct1 = place_adsorbate(surface=sub1, adsorbate=mol)
    mol = molecule("H")
    base_struct2 = place_adsorbate(surface=sub2, adsorbate=mol)
    base_struct1[-2].tag = 1
    p_set = generate_perturbed_dataset(
        [base_struct1, base_struct2], num_of_perturbations=5,
    )
    manual_collect = []
    for base in ["HOPt36_0", "HCu36_1"]:
        for struct in p_set[base]:
            manual_collect.append(p_set[base][struct]["perturbation_matrix"].flatten())
    manual_collect_array = np.zeros((10, 3))
    for idx, m in enumerate(manual_collect):
        manual_collect_array[idx, : len(m)] = m
    assert np.allclose(p_set["correction_matrix"], manual_collect_array)
    assert p_set["correction_matrix"].shape == (10, 3)
    # check corrections list
    corr_list = p_set["corrections_list"]
    assert len(corr_list) == p_set["correction_matrix"].shape[0]
    assert (corr_list[3] == p_set["correction_matrix"][3, : len(corr_list[3][0])]).all()


def test_generate_perturbed_dataset_collected_structure_paths():
    # Tests that collected structure paths in correct order
    _tmp_dir = tempfile.TemporaryDirectory().name
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct = place_adsorbate(surface=sub, adsorbate=mol)
    p_set = generate_perturbed_dataset(
        [base_struct], write_to_disk=True, write_location=_tmp_dir,
    )
    assert os.path.samefile(
        p_set["collected_structure_paths"][0],
        os.path.join(_tmp_dir, "HOPt36_0/0/perturbed_structure.traj"),
    )
    assert os.path.samefile(
        p_set["collected_structure_paths"][7],
        os.path.join(_tmp_dir, "HOPt36_0/7/perturbed_structure.traj"),
    )


def test_generate_perturbed_dataset_collected_structures():
    # Test that all of the structures are collected
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    mol = molecule("OH")
    base_struct1 = place_adsorbate(surface=sub1, adsorbate=mol)
    base_struct2 = place_adsorbate(surface=sub2, adsorbate=mol)
    base_struct1[-2].tag = 1
    base_struct2[-1].tag = 1
    p_set = generate_perturbed_dataset(
        [base_struct1, base_struct2], num_of_perturbations=5,
    )
    assert len(p_set["collected_structures"]) == 10
    assert isinstance(p_set["collected_structures"][3], Atoms)
