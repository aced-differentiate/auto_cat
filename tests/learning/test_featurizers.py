"""Unit tests for the `autocat.learning.featurizersi` module."""

import os
import numpy as np
import json

import pytest

import tempfile

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

import qml

from autocat.io.qml import ase_atoms_to_qml_compound
from autocat.adsorption import generate_rxn_structures
from autocat.surface import generate_surface_structures
from autocat.learning.featurizers import full_structure_featurization
from autocat.learning.featurizers import adsorbate_featurization
from autocat.learning.featurizers import catalyst_featurization
from autocat.learning.featurizers import _get_number_of_features
from autocat.learning.featurizers import get_X


def test_full_structure_featurization_sine():
    # Tests Sine Matrix Generation
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    sine_matrix = full_structure_featurization(surf)
    sm = SineMatrix(n_atoms_max=len(surf), permutation="none")
    assert np.allclose(sine_matrix, sm.create(surf))
    # Check padding
    sine_matrix = full_structure_featurization(surf, maximum_structure_size=40)
    assert sine_matrix.shape == (1600,)


def test_full_structure_featurization_coulomb():
    # Tests Coulomb Matrix Generation
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    coulomb_matrix = full_structure_featurization(surf, featurizer="coulomb_matrix")
    cm = CoulombMatrix(n_atoms_max=len(surf), permutation="none")
    assert np.allclose(coulomb_matrix, cm.create(surf))
    # Check padding
    coulomb_matrix = full_structure_featurization(surf, maximum_structure_size=45)
    assert coulomb_matrix.shape == (2025,)


def test_adsorbate_featurization_acsf():
    # Tests Atom Centered Symmetry Function generation
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    acsf_feat = adsorbate_featurization(ads_struct, [36], featurizer="acsf")
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    acsf = ACSF(rcut=6.0, species=species)
    assert np.allclose(acsf_feat, acsf.create(ads_struct, [36]))


def test_adsorbate_featurization_soap():
    # Tests Smooth Overlap of Atomic Positions
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    soap_feat = adsorbate_featurization(
        ads_struct, [-1], featurizer="soap", nmax=8, lmax=6
    )
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    soap = SOAP(rcut=6.0, species=species, nmax=8, lmax=6)
    assert np.allclose(soap_feat, soap.create(ads_struct, [-1]))
    assert soap_feat.shape == (soap.get_number_of_features(),)


def test_adsorbate_featurization_padding():
    # Tests that padding is properly applied
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    soap_feat = adsorbate_featurization(
        ads_struct, [36], featurizer="soap", nmax=8, lmax=6, maximum_adsorbate_size=10
    )

    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    num_of_features = _get_number_of_features(
        featurizer="soap", species=species, rcut=6.0, nmax=8, lmax=6
    )

    assert (soap_feat[-num_of_features * 9 :] == np.zeros(num_of_features * 9)).all()
    soap_feat = adsorbate_featurization(
        ads_struct,
        [35, 36],
        featurizer="soap",
        nmax=8,
        lmax=6,
        maximum_adsorbate_size=10,
    )
    assert (soap_feat[-num_of_features * 8 :] == np.zeros(num_of_features * 8)).all()


def test_catalyst_featurization_concatentation():
    # Tests that the representations are properly concatenated
    # with kwargs input appropriately
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["OH"])["OH"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    cat = catalyst_featurization(
        ads_struct,
        [-1, -2],
        maximum_structure_size=40,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    sm = SineMatrix(n_atoms_max=40, permutation="none")
    struct = sm.create(ads_struct).reshape(-1,)
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    soap = SOAP(rcut=5.0, nmax=8, lmax=6, species=species)
    ads = soap.create(ads_struct, [-1, -2]).reshape(-1,)
    cat_ref = np.concatenate((struct, ads))
    assert np.allclose(cat, cat_ref)
    num_of_adsorbate_features = soap.get_number_of_features()
    assert len(cat) == 40 ** 2 + num_of_adsorbate_features * 2


def test_get_X_concatenation():
    # Tests that the resulting X is concatenated and ordered properly
    structs = []
    surf1 = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads1 = generate_rxn_structures(
        surf1,
        ads=["NH3", "CO"],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)]},
        height={"CO": 1.5},
        rots={"NH3": [[180.0, "x"], [90.0, "z"]], "CO": [[180.0, "y"]]},
    )
    structs.append(ads1["NH3"]["origin"]["0.0_0.0"]["structure"])
    structs.append(ads1["CO"]["origin"]["0.0_0.0"]["structure"])
    surf2 = generate_surface_structures(["Ru"])["Ru"]["hcp0001"]["structure"]
    ads2 = generate_rxn_structures(
        surf2, ads=["N"], all_sym_sites=False, sites={"origin": [(0.0, 0.0)]},
    )
    structs.append(ads2["N"]["origin"]["0.0_0.0"]["structure"])

    X = get_X(
        structs,
        adsorbate_indices_dictionary={
            structs[0].get_chemical_formula(): [-4, -3, -2, -1],
            structs[1].get_chemical_formula(): [-2, -1],
            structs[2].get_chemical_formula(): [-1],
        },
        maximum_structure_size=50,
        maximum_adsorbate_size=5,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    species_list = ["Pt", "Ru", "N", "C", "O", "H"]
    num_of_adsorbate_features = _get_number_of_features(
        featurizer="soap", rcut=5.0, nmax=8, lmax=6, species=species_list
    )
    assert X.shape == (len(structs), 50 ** 2 + 5 * num_of_adsorbate_features)


def test_get_X_write_location():
    # Tests user-specified write location for X
    structs = []
    surf1 = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads1 = generate_rxn_structures(
        surf1,
        ads=["NH3", "CO"],
        all_sym_sites=False,
        sites={"origin": [(0.0, 0.0)]},
        height={"CO": 1.5},
        rots={"NH3": [[180.0, "x"], [90.0, "z"]], "CO": [[180.0, "y"]]},
    )
    structs.append(ads1["NH3"]["origin"]["0.0_0.0"]["structure"])
    structs.append(ads1["CO"]["origin"]["0.0_0.0"]["structure"])

    _tmp_dir = tempfile.TemporaryDirectory().name
    X = get_X(
        structs,
        adsorbate_indices_dictionary={
            structs[0].get_chemical_formula(): [-4, -3, -2, -1],
            structs[1].get_chemical_formula(): [-2, -1],
        },
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    with open(os.path.join(_tmp_dir, "X.json"), "r") as f:
        X_written = json.load(f)
        assert np.allclose(X, X_written)
