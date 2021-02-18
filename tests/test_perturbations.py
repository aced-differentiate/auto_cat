"""Unit tests for the `autocat.perturbations` module"""

import os
import pytest
import numpy as np

import tempfile

from autocat.surface import generate_surface_structures
from autocat.adsorption import place_adsorbate
from autocat.perturbations import perturb_structure
from autocat.perturbations import generate_perturbed_dataset


def test_perturb_structure_directions():
    # Tests fixing directions of perturbations
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "H")["custom"]["structure"]
    # fixed in all directions
    p_struct = perturb_structure(
        base_struct, atom_indices_to_perturb=[-1], directions=[False, False, False]
    )
    assert (p_struct["structure"].positions[-1] == base_struct.positions[-1]).all()
    # free in all directions
    p_struct = perturb_structure(
        base_struct, atom_indices_to_perturb=[-1], directions=[True, True, True]
    )
    assert (p_struct["structure"].positions[-1] != base_struct.positions[-1]).all()
    # free in 1D
    p_struct = perturb_structure(
        base_struct, atom_indices_to_perturb=[-1], directions=[False, False, True]
    )
    assert p_struct["structure"].positions[-1][0] == base_struct.positions[-1][0]
    assert p_struct["structure"].positions[-1][1] == base_struct.positions[-1][1]
    assert p_struct["structure"].positions[-1][-1] != base_struct.positions[-1][-1]


def test_perturb_structure_matrix():
    # Tests matrix matches perturbed structure
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "OH")["custom"]["structure"]
    p_struct = perturb_structure(base_struct, atom_indices_to_perturb=[-1, -2])
    o_pert = base_struct.positions[-2] + p_struct["perturbation_matrix"][-2]
    assert np.allclose(p_struct["structure"].positions[-2], o_pert)
    h_pert = base_struct.positions[-1] + p_struct["perturbation_matrix"][-1]
    assert np.allclose(p_struct["structure"].positions[-1], h_pert)


def test_generate_perturbed_dataset_num_of_perturbations():
    # Tests number of perturbations generated
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "OH")["custom"]["structure"]
    p_set = generate_perturbed_dataset(
        [base_struct],
        atom_indices_to_perturb_dictionary={
            base_struct.get_chemical_formula(): [-1, -2]
        },
        num_of_perturbations=15,
    )
    assert len(p_set["HOPt36"].keys()) == 15


def test_generate_perturbed_dataset_multiple_base_structures():
    # Tests giving multiple base_structures
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    base_struct2 = place_adsorbate(sub2, "NH")["custom"]["structure"]
    p_set = generate_perturbed_dataset(
        [base_struct1, base_struct2],
        atom_indices_to_perturb_dictionary={
            base_struct1.get_chemical_formula(): [-1],
            base_struct2.get_chemical_formula(): [-2],
        },
        directions_dictionary={
            base_struct1.get_chemical_formula(): [False, False, True]
        },
    )
    # Check all base structures perturbed
    assert len(p_set.keys()) - 1 == 2
    assert "HCu36N" in p_set
    assert "HOPt36" in p_set
    # Check correct atom indices perturbed for each base_structure
    assert (p_set["HCu36N"]["2"]["perturbation_matrix"][-1] == np.zeros(3)).all()
    assert (p_set["HCu36N"]["3"]["perturbation_matrix"][-2] != np.zeros(3)).all()
    assert (p_set["HOPt36"]["6"]["perturbation_matrix"][-1] != np.zeros(3)).any()
    # Check correct direction constraints applied to each base_structure
    assert np.isclose(p_set["HOPt36"]["6"]["perturbation_matrix"][-1][0], 0.0)
    assert np.isclose(p_set["HOPt36"]["6"]["perturbation_matrix"][-1][1], 0.0)
    assert not np.isclose(p_set["HOPt36"]["6"]["perturbation_matrix"][-1][-1], 0.0)
    assert not np.isclose(p_set["HCu36N"]["1"]["perturbation_matrix"][-2][0], 0.0)


def test_generate_perturbed_dataset_write_location():
    # Tests write location
    _tmp_dir = tempfile.TemporaryDirectory().name
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "OH")["custom"]["structure"]
    p_set = generate_perturbed_dataset(
        [base_struct],
        atom_indices_to_perturb_dictionary={base_struct.get_chemical_formula(): [-1]},
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    assert os.path.samefile(
        p_set["HOPt36"]["0"]["traj_file_path"],
        os.path.join(_tmp_dir, "HOPt36/0/perturbed_structure.traj"),
    )
    assert os.path.samefile(
        p_set["HOPt36"]["0"]["pert_mat_file_path"],
        os.path.join(_tmp_dir, "HOPt36/0/perturbation_matrix.json"),
    )


def test_generate_perturbed_dataset_collected_matrices():
    # Check that matrices properly collected for 1 base_structure
    sub = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    base_struct = place_adsorbate(sub, "OH")["custom"]["structure"]
    p_set = generate_perturbed_dataset(
        [base_struct],
        atom_indices_to_perturb_dictionary={
            base_struct.get_chemical_formula(): [-1, -2]
        },
        num_of_perturbations=5,
    )
    manual_collect = []
    for struct in p_set["HOPt36"]:
        manual_collect.append(p_set["HOPt36"][struct]["perturbation_matrix"].flatten())
    manual_collect = np.array(manual_collect)
    assert np.allclose(p_set["collected_matrices"], manual_collect)
    assert p_set["collected_matrices"].shape == (5, 114)


def test_generate_perturbed_dataset_collected_matrices_multiple():
    # check that matrices properly collected for 2 base structures
    sub1 = generate_surface_structures(["Pt"], facets={"Pt": ["111"]})["Pt"]["fcc111"][
        "structure"
    ]
    sub2 = generate_surface_structures(["Cu"], facets={"Cu": ["100"]})["Cu"]["fcc100"][
        "structure"
    ]
    base_struct1 = place_adsorbate(sub1, "OH")["custom"]["structure"]
    base_struct2 = place_adsorbate(sub2, "OH")["custom"]["structure"]
    p_set = generate_perturbed_dataset(
        [base_struct1, base_struct2],
        atom_indices_to_perturb_dictionary={
            base_struct1.get_chemical_formula(): [-1],
            base_struct2.get_chemical_formula(): [-2],
        },
        num_of_perturbations=5,
    )
    manual_collect = []
    for base in ["HOPt36", "HCu36O"]:
        for struct in p_set[base]:
            manual_collect.append(p_set[base][struct]["perturbation_matrix"].flatten())
    manual_collect = np.array(manual_collect)
    assert np.allclose(p_set["collected_matrices"], manual_collect)
    assert p_set["collected_matrices"].shape == (10, 114)
