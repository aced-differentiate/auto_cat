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

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import ChemicalSRO
from matminer.featurizers.site import OPSiteFingerprint
from matminer.featurizers.site import CrystalNNFingerprint

# from autocat.io.qml import ase_atoms_to_qml_compound
from autocat.adsorption import generate_rxn_structures
from autocat.surface import generate_surface_structures
from autocat.learning.featurizers import full_structure_featurization
from autocat.learning.featurizers import adsorbate_featurization
from autocat.learning.featurizers import catalyst_featurization
from autocat.learning.featurizers import _get_number_of_features
from autocat.learning.featurizers import get_X

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN


def test_full_structure_featurization_sine():
    # Tests Sine Matrix Generation
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    sine_matrix = full_structure_featurization(surf, refine_structure=False)
    sm = SineMatrix(n_atoms_max=len(surf), permutation="none")
    assert np.allclose(sine_matrix, sm.create(surf))
    # Check padding
    sine_matrix = full_structure_featurization(surf, maximum_structure_size=40)
    assert sine_matrix.shape == (1600,)
    # Check refined structure
    sine_matrix = full_structure_featurization(surf, refine_structure=True)
    surf = surf[np.where(surf.get_tags() < 2)[0].tolist()]
    sm = SineMatrix(n_atoms_max=len(surf), permutation="none")
    assert np.allclose(sine_matrix, sm.create(surf))
    assert sine_matrix.shape == (len(surf) ** 2,)


def test_full_structure_featurization_coulomb():
    # Tests Coulomb Matrix Generation
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc100"]["structure"]
    coulomb_matrix = full_structure_featurization(
        surf, featurizer="coulomb_matrix", refine_structure=False
    )
    cm = CoulombMatrix(n_atoms_max=len(surf), permutation="none")
    assert np.allclose(coulomb_matrix, cm.create(surf))
    # Check padding
    coulomb_matrix = full_structure_featurization(surf, maximum_structure_size=45)
    assert coulomb_matrix.shape == (2025,)


def test_full_structure_featurization_elemental_property():
    # Tests the Elemental Property featurization
    surf = generate_surface_structures(["Cu"])["Cu"]["fcc111"]["structure"]
    elem_prop = full_structure_featurization(surf, featurizer="elemental_property")
    ep = ElementProperty.from_preset("magpie")
    conv = AseAtomsAdaptor()
    pymat = conv.get_structure(surf)
    manual_elem_prop = ep.featurize(pymat.composition)
    assert np.allclose(elem_prop, manual_elem_prop)
    assert elem_prop.shape == (132,)
    elem_prop = full_structure_featurization(
        surf, featurizer="elemental_property", elementalproperty_preset="deml"
    )
    ep = ElementProperty.from_preset("deml")
    manual_elem_prop = ep.featurize(pymat.composition)
    assert np.allclose(elem_prop, manual_elem_prop)
    assert len(elem_prop) == _get_number_of_features("elemental_property", "deml")


def test_adsorbate_featurization_acsf():
    # Tests Atom Centered Symmetry Function generation
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    acsf_feat = adsorbate_featurization(
        ads_struct, featurizer="acsf", refine_structure=False
    )
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    acsf = ACSF(rcut=6.0, species=species)
    assert np.allclose(acsf_feat, acsf.create(ads_struct, [36]))


def test_adsorbate_featurization_soap():
    # Tests Smooth Overlap of Atomic Positions
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    soap_feat = adsorbate_featurization(ads_struct, featurizer="soap", nmax=8, lmax=6)
    ads_struct = ads_struct[np.where(ads_struct.get_tags() < 2)[0].tolist()]
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    soap = SOAP(rcut=6.0, species=species, nmax=8, lmax=6)
    assert np.allclose(soap_feat, soap.create(ads_struct, positions=[-1]))
    assert soap_feat.shape == (soap.get_number_of_features(),)


def test_adsorbate_featurization_chemical_sro():
    # Tests Chemical Short Range Ordering Featurization
    surf = generate_surface_structures(["Li"])["Li"]["bcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["OH"])["OH"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    csro_feat = adsorbate_featurization(
        ads_struct,
        featurizer="chemical_sro",
        rcut=10.0,
        species_list=["Li", "O", "H"],
        refine_structure=False,
    )
    assert csro_feat.shape == (6,)
    species = ["Li", "O", "H"]
    vnn = VoronoiNN(cutoff=10.0, allow_pathological=True)
    csro = ChemicalSRO(vnn, includes=species)
    conv = AseAtomsAdaptor()
    pym_struct = conv.get_structure(ads_struct)
    csro.fit([[pym_struct, -2], [pym_struct, -1]])
    manual_feat = csro.featurize(pym_struct, -2)
    manual_feat = np.concatenate((manual_feat, csro.featurize(pym_struct, -1)))
    assert np.allclose(csro_feat, manual_feat)


def test_adsorbate_featurization_op_sitefingerprint():
    # Test Order Parameter Site Fingerprints
    surf = generate_surface_structures(["Au"])["Au"]["fcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["NH"])["NH"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    opsf_feat = adsorbate_featurization(
        ads_struct,
        featurizer="op_sitefingerprint",
        maximum_adsorbate_size=4,
        refine_structure=False,
    )
    opsf = OPSiteFingerprint()
    conv = AseAtomsAdaptor()
    pym_struct = conv.get_structure(ads_struct)
    manual_feat = opsf.featurize(pym_struct, -2)
    manual_feat = np.concatenate((manual_feat, opsf.featurize(pym_struct, -1)))
    manual_feat = np.concatenate(
        (manual_feat, np.zeros(2 * len(opsf.feature_labels())))
    )
    assert np.allclose(opsf_feat, manual_feat)


def test_adsorbate_featurization_crystalnn_fingerprint():
    # Test CrystalNN site fingerprint
    surf = generate_surface_structures(["Ag"])["Ag"]["fcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["OH"])["OH"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    cnn_feat = adsorbate_featurization(
        ads_struct,
        featurizer="crystalnn_sitefingerprint",
        maximum_adsorbate_size=4,
        refine_structure=False,
    )
    cnn = CrystalNNFingerprint.from_preset("cn")
    conv = AseAtomsAdaptor()
    pym_struct = conv.get_structure(ads_struct)
    manual_feat = cnn.featurize(pym_struct, -2)
    manual_feat = np.concatenate((manual_feat, cnn.featurize(pym_struct, -1)))
    manual_feat = np.concatenate((manual_feat, np.zeros(2 * len(cnn.feature_labels()))))
    assert np.allclose(cnn_feat, manual_feat)
    assert cnn_feat.shape[0] == 4 * len(cnn.feature_labels())


def test_adsorbate_featurization_padding():
    # Tests that padding is properly applied
    surf = generate_surface_structures(["Fe"])["Fe"]["bcc100"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["H"])["H"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    soap_feat = adsorbate_featurization(
        ads_struct, featurizer="soap", nmax=8, lmax=6, maximum_adsorbate_size=10
    )

    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    num_of_features = _get_number_of_features(
        featurizer="soap", species=species, rcut=6.0, nmax=8, lmax=6
    )

    assert (soap_feat[-num_of_features * 9 :] == np.zeros(num_of_features * 9)).all()
    ads_struct[35].tag = 0
    soap_feat = adsorbate_featurization(
        ads_struct, featurizer="soap", nmax=8, lmax=6, maximum_adsorbate_size=10,
    )
    assert (soap_feat[-num_of_features * 8 :] == np.zeros(num_of_features * 8)).all()
    csro_feat = adsorbate_featurization(
        ads_struct,
        featurizer="chemical_sro",
        rcut=10.0,
        maximum_adsorbate_size=4,
        species_list=["Fe", "H", "Li"],
    )
    assert csro_feat.shape == (12,)
    assert (csro_feat[6:] == np.zeros(6)).all()
    assert csro_feat[5] == 0.0
    assert csro_feat[2] == 0.0


def test_catalyst_featurization_concatentation():
    # Tests that the representations are properly concatenated
    # with kwargs input appropriately
    surf = generate_surface_structures(["Pt"])["Pt"]["fcc111"]["structure"]
    ads_struct = generate_rxn_structures(surf, ads=["OH"])["OH"]["ontop"]["0.0_0.0"][
        "structure"
    ]
    cat = catalyst_featurization(
        ads_struct,
        maximum_structure_size=40,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
        refine_structure=False,
    )
    sm = SineMatrix(n_atoms_max=40, permutation="none")
    struct = sm.create(ads_struct).reshape(-1,)
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    soap = SOAP(rcut=5.0, nmax=8, lmax=6, species=species)
    ads = soap.create(ads_struct, [-2, -1]).reshape(-1,)
    cat_ref = np.concatenate((struct, ads))
    assert np.allclose(cat, cat_ref)
    num_of_adsorbate_features = soap.get_number_of_features()
    assert len(cat) == 40 ** 2 + num_of_adsorbate_features * 2
    # Check with structure refining
    cat = catalyst_featurization(
        ads_struct, adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    ref_ads_struct = ads_struct[np.where(ads_struct.get_tags() < 2)[0].tolist()]
    sm = SineMatrix(n_atoms_max=len(ref_ads_struct), permutation="none")
    num_of_adsorbate_features = soap.get_number_of_features()
    assert len(cat) == len(ref_ads_struct) ** 2 + num_of_adsorbate_features * 2


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
        maximum_structure_size=50,
        maximum_adsorbate_size=5,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    species_list = ["Pt", "Ru", "N", "C", "O", "H"]
    num_of_adsorbate_features = _get_number_of_features(
        featurizer="soap", rcut=5.0, nmax=8, lmax=6, species=species_list
    )
    assert X.shape == (len(structs), 50 ** 2 + 5 * num_of_adsorbate_features)
    # Check for full structure featurization only
    X = get_X(
        structs,
        structure_featurizer="elemental_property",
        adsorbate_featurizer=None,
        maximum_structure_size=50,
        maximum_adsorbate_size=5,
    )
    assert X.shape == (len(structs), 132)
    # Check for adsorbate featurization only
    X = get_X(
        structs,
        structure_featurizer=None,
        maximum_structure_size=50,
        maximum_adsorbate_size=5,
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
    )
    species_list = ["Pt", "Ru", "N", "C", "O", "H"]
    num_of_adsorbate_features = _get_number_of_features(
        featurizer="soap", rcut=5.0, nmax=8, lmax=6, species=species_list
    )
    assert X.shape == (len(structs), 5 * num_of_adsorbate_features)
    X = get_X(
        structs,
        structure_featurizer=None,
        adsorbate_featurizer="chemical_sro",
        maximum_adsorbate_size=5,
        species_list=species_list,
    )
    num_of_adsorbate_features = _get_number_of_features(
        featurizer="chemical_sro", rcut=5.0, species=species_list
    )
    assert X.shape == (len(structs), 5 * num_of_adsorbate_features)
    X = get_X(
        structs,
        structure_featurizer=None,
        adsorbate_featurizer="op_sitefingerprint",
        maximum_adsorbate_size=5,
        species_list=species_list,
    )
    num_of_adsorbate_features = _get_number_of_features(featurizer="op_sitefingerprint")
    assert X.shape == (len(structs), 5 * num_of_adsorbate_features)


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
        adsorbate_featurization_kwargs={"rcut": 5.0, "nmax": 8, "lmax": 6},
        write_to_disk=True,
        write_location=_tmp_dir,
    )
    with open(os.path.join(_tmp_dir, "X.json"), "r") as f:
        X_written = json.load(f)
        assert np.allclose(X, X_written)
