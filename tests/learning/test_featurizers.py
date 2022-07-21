"""Unit tests for the `autocat.learning.featurizers` module."""

import os
import tempfile

import numpy as np
import pytest

from ase import Atoms
from ase.io.jsonio import encode as atoms_encoder

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import ChemicalSRO
from matminer.featurizers.site import OPSiteFingerprint
from matminer.featurizers.site import CrystalNNFingerprint

from autocat.adsorption import generate_adsorbed_structures
from autocat.surface import generate_surface_structures
from autocat.saa import generate_saa_structures
from autocat.learning.featurizers import Featurizer, FeaturizerError
from autocat.utils import flatten_structures_dict

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN


def test_featurizer_copy():
    # test making a copy
    f = Featurizer(
        SOAP,
        max_size=5,
        species_list=["Fe", "O", "H"],
        kwargs={"rcut": 12, "nmax": 8, "lmax": 8},
    )
    f2 = f.copy()
    assert f == f2
    assert f is not f2

    f = Featurizer(
        ElementProperty,
        preset="matminer",
        design_space_structures=[Atoms("H2"), Atoms("N2")],
    )
    f2 = f.copy()
    assert f == f2
    assert f is not f2


def test_eq_featurizer():
    # test comparing featurizers

    f = Featurizer(
        SOAP,
        max_size=5,
        species_list=["Fe", "O", "H"],
        kwargs={"rcut": 12, "nmax": 8, "lmax": 8},
    )
    f1 = Featurizer(
        SOAP,
        max_size=5,
        species_list=["Fe", "O", "H"],
        kwargs={"rcut": 12, "nmax": 8, "lmax": 8},
    )
    assert f == f1

    f1.kwargs.update({"rcut": 13})
    assert f != f1

    surfs = flatten_structures_dict(generate_surface_structures(["Fe", "V"]))
    surfs.extend(
        flatten_structures_dict(
            generate_surface_structures(["Au", "Ag"], supercell_dim=(1, 1, 5))
        )
    )
    f = Featurizer(SineMatrix, design_space_structures=surfs,)

    f1 = Featurizer(SineMatrix, species_list=["Fe", "V", "Au", "Ag"], max_size=36)
    assert f == f1


def test_featurizer_species_list():
    # test default species list
    f = Featurizer(SineMatrix)
    assert f.species_list == ["Fe", "Ni", "Pt", "Pd", "Cu", "C", "N", "O", "H"]

    # test updating species list manually and sorting
    f.species_list = ["Li", "Na", "K"]
    assert f.species_list == ["K", "Na", "Li"]

    # test getting species list from design space structures
    surfs = flatten_structures_dict(generate_surface_structures(["Fe", "V", "Ti"]))
    saas = flatten_structures_dict(generate_saa_structures(["Cu", "Au"], ["Fe", "Pt"]))
    surfs.extend(saas)
    f.design_space_structures = surfs
    assert f.species_list == ["Ti", "V", "Fe", "Pt", "Au", "Cu"]


def test_featurizer_max_size():
    # test default max size
    f = Featurizer(SOAP, kwargs={"rcut": 12, "nmax": 8, "lmax": 8})
    assert f.max_size == 100

    # test updating max size manually
    f.max_size = 50
    assert f.max_size == 50

    # test getting max size from design space structures
    surfs = flatten_structures_dict(
        generate_surface_structures(["Ru"], supercell_dim=(2, 2, 4))
    )
    surfs.extend(
        flatten_structures_dict(
            generate_surface_structures(["Fe"], supercell_dim=(4, 4, 4))
        )
    )
    f.design_space_structures = surfs
    assert f.max_size == 64


def test_featurizer_design_space_structures():
    # tests giving design space structures
    surfs = flatten_structures_dict(generate_surface_structures(["Li", "Na"]))
    surfs.extend(
        flatten_structures_dict(
            generate_surface_structures(["Cu", "Ni"], supercell_dim=(1, 1, 5))
        )
    )
    f = Featurizer(
        SineMatrix, design_space_structures=surfs, max_size=20, species_list=["H"]
    )
    assert f.design_space_structures == surfs
    # make sure design space is prioritized over max size and species list
    assert f.max_size == 36
    assert f.species_list == ["Na", "Li", "Ni", "Cu"]


def test_featurizer_featurizer_kwargs():
    # test specifying kwargs
    f = Featurizer(CoulombMatrix, kwargs={"flatten": False})
    assert f.kwargs == {"flatten": False}
    assert f.featurization_object.flatten == False

    # test updating kwargs
    f.kwargs.update({"sparse": True})
    assert f.featurization_object.sparse == True

    # test rm kwargs when updating class
    f.featurizer_class = SineMatrix
    assert f.kwargs is None


def test_featurizer_featurizer_class():
    # test changing featurizer class
    f = Featurizer(SOAP, kwargs={"rcut": 12, "nmax": 8, "lmax": 8})
    assert f.featurizer_class == SOAP
    assert isinstance(f.featurization_object, SOAP)
    f.featurizer_class = SineMatrix
    assert f.featurizer_class == SineMatrix
    assert isinstance(f.featurization_object, SineMatrix)


def test_featurizer_preset():
    # tests specifying preset for class object
    f = Featurizer(ElementProperty, preset="magpie")
    assert f.preset == "magpie"
    assert "Electronegativity" in f.featurization_object.features
    assert not "melting_point" in f.featurization_object.features

    f.preset = "matminer"
    assert f.preset == "matminer"
    assert not "NdUnfilled" in f.featurization_object.features
    assert "coefficient_of_linear_thermal_expansion" in f.featurization_object.features


def test_featurizer_featurize_single():
    # tests featurizing single structure at a time

    conv = AseAtomsAdaptor()

    # TEST STRUCTURE FEATURIZERS

    # test ElementProperty
    saa = flatten_structures_dict(generate_saa_structures(["Cu"], ["Pt"]))[0]
    f = Featurizer(ElementProperty, preset="magpie", max_size=len(saa))
    acf = f.featurize_single(saa)
    ep = ElementProperty.from_preset("magpie")
    pymat = conv.get_structure(saa)
    manual_elem_prop = ep.featurize(pymat.composition)
    assert np.array_equal(acf, manual_elem_prop)

    # test SineMatrix
    f.featurizer_class = SineMatrix
    acf = f.featurize_single(saa)
    sm = SineMatrix(n_atoms_max=len(saa), permutation="none")
    manual_sm = sm.create(saa).reshape(-1,)
    assert np.array_equal(acf, manual_sm)

    # test CoulombMatrix
    f.featurizer_class = CoulombMatrix
    acf = f.featurize_single(saa)
    cm = CoulombMatrix(n_atoms_max=len(saa), permutation="none")
    manual_cm = cm.create(saa).reshape(-1,)
    assert np.array_equal(acf, manual_cm)

    # TEST SITE FEATURIZERS
    ads_struct = flatten_structures_dict(
        generate_adsorbed_structures(
            surface=saa,
            adsorbates=["OH"],
            adsorption_sites={"custom": [(0.0, 0.0)]},
            use_all_sites=False,
        )
    )[0]
    f.max_size = len(ads_struct)
    species = np.unique(ads_struct.get_chemical_symbols()).tolist()
    f.species_list = species

    # test ACSF
    f.featurizer_class = ACSF
    f.kwargs = {"rcut": 6.0}
    acf = f.featurize_single(ads_struct)
    acsf = ACSF(rcut=6.0, species=species)
    manual_acsf = acsf.create(ads_struct, [36, 37])
    assert np.array_equal(acf, manual_acsf)

    # test SOAP
    f.featurizer_class = SOAP
    f.kwargs = {"rcut": 6.0, "lmax": 6, "nmax": 6}
    acf = f.featurize_single(ads_struct)
    soap = SOAP(rcut=6.0, species=species, nmax=6, lmax=6)
    manual_soap = soap.create(ads_struct, [36, 37])
    assert np.array_equal(acf, manual_soap)

    # test ChemicalSRO
    f.featurizer_class = ChemicalSRO
    vnn = VoronoiNN(cutoff=10.0, allow_pathological=True)
    f.kwargs = {"nn": vnn, "includes": species}
    acf = f.featurize_single(ads_struct)
    csro = ChemicalSRO(vnn, includes=species)
    pym_struct = conv.get_structure(ads_struct)
    csro.fit([[pym_struct, 36], [pym_struct, 37]])
    manual_csro = csro.featurize(pym_struct, -2)
    manual_csro = np.concatenate((manual_csro, csro.featurize(pym_struct, -1)))
    assert np.array_equal(acf, manual_csro)

    # test OPSiteFingerprint
    f.featurizer_class = OPSiteFingerprint
    acf = f.featurize_single(ads_struct)
    pym_struct = conv.get_structure(ads_struct)
    opsf = OPSiteFingerprint()
    manual_opsf = opsf.featurize(pym_struct, -2)
    manual_opsf = np.concatenate((manual_opsf, opsf.featurize(pym_struct, -1)))
    assert np.array_equal(acf, manual_opsf)

    # test CrystalNNFingerprint
    f.featurizer_class = CrystalNNFingerprint
    f.preset = "cn"
    acf = f.featurize_single(ads_struct)
    pym_struct = conv.get_structure(ads_struct)
    cnn = CrystalNNFingerprint.from_preset("cn")
    manual_cnn = cnn.featurize(pym_struct, -2)
    manual_cnn = np.concatenate((manual_cnn, cnn.featurize(pym_struct, -1)))
    assert np.array_equal(acf, manual_cnn)


def test_featurizer_featurize_multiple():
    # tests featurizing multiple structures at a time

    # TEST STRUCTURE FEATURIZER

    # test ElementProperty
    saas = flatten_structures_dict(
        generate_saa_structures(
            ["Au", "Cu"], ["Pd", "Pt"], facets={"Au": ["111"], "Cu": ["111"]}
        )
    )
    f = Featurizer(ElementProperty, preset="magpie", design_space_structures=saas)
    acf = f.featurize_multiple(saas)
    manual_mat = []
    for i in range(len(saas)):
        manual_mat.append(f.featurize_single(saas[i]))
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)

    # test SineMatrix
    f.featurizer_class = SineMatrix
    acf = f.featurize_multiple(saas)
    manual_mat = []
    for i in range(len(saas)):
        manual_mat.append(f.featurize_single(saas[i]))
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)

    # test CoulombMatrix
    f.featurizer_class = CoulombMatrix
    acf = f.featurize_multiple(saas)
    manual_mat = []
    for i in range(len(saas)):
        manual_mat.append(f.featurize_single(saas[i]))
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)

    # TEST SITE FEATURIZER
    ads_structs = []
    for struct in saas:
        ads_structs.append(
            flatten_structures_dict(
                generate_adsorbed_structures(
                    surface=struct,
                    adsorbates=["NNH"],
                    adsorption_sites={"custom": [(0.0, 0.0)]},
                    use_all_sites=False,
                )
            )[0]
        )
    species_list = []
    for s in ads_structs:
        # get all unique species
        found_species = np.unique(s.get_chemical_symbols()).tolist()
        new_species = [spec for spec in found_species if spec not in species_list]
        species_list.extend(new_species)

    # test SOAP
    f.featurizer_class = SOAP
    f.design_space_structures = ads_structs
    f.kwargs = {"rcut": 6.0, "lmax": 6, "nmax": 6}
    acf = f.featurize_multiple(ads_structs)
    manual_mat = []
    for i in range(len(ads_structs)):
        manual_mat.append(f.featurize_single(ads_structs[i]).flatten())
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)

    # test ACSF
    f.featurizer_class = ACSF
    f.kwargs = {"rcut": 6.0}
    acf = f.featurize_multiple(ads_structs)
    manual_mat = []
    for i in range(len(ads_structs)):
        manual_mat.append(f.featurize_single(ads_structs[i]).flatten())
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)

    # test ChemicalSRO
    f.featurizer_class = ChemicalSRO
    vnn = VoronoiNN(cutoff=10.0, allow_pathological=True)
    f.kwargs = {"nn": vnn, "includes": species_list}
    acf = f.featurize_multiple(ads_structs)
    manual_mat = []
    for i in range(len(ads_structs)):
        manual_mat.append(f.featurize_single(ads_structs[i]).flatten())
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)

    # test OPSiteFingerprint
    f.featurizer_class = OPSiteFingerprint
    acf = f.featurize_multiple(ads_structs)
    manual_mat = []
    for i in range(len(ads_structs)):
        manual_mat.append(f.featurize_single(ads_structs[i]).flatten())
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)

    # test CrystalNNFingerprint
    f.featurizer_class = CrystalNNFingerprint
    f.preset = "cn"
    acf = f.featurize_multiple(ads_structs)
    manual_mat = []
    for i in range(len(ads_structs)):
        manual_mat.append(f.featurize_single(ads_structs[i]).flatten())
    manual_mat = np.array(manual_mat)
    assert np.array_equal(acf, manual_mat)


def test_featurizer_from_json():
    # Tests generating a Featurizer from a json
    surfs = flatten_structures_dict(generate_surface_structures(["Fe", "V"]))
    surfs.extend(
        flatten_structures_dict(
            generate_surface_structures(["Au", "Ag"], supercell_dim=(1, 1, 5))
        )
    )
    f = Featurizer(SineMatrix, design_space_structures=surfs,)
    with tempfile.TemporaryDirectory() as _tmp_dir:
        f.write_json_to_disk(write_location=_tmp_dir, json_name="test_feat.json")
        json_path = os.path.join(_tmp_dir, "test_feat.json")
        written_f = Featurizer.from_json(json_path)
        assert written_f == f

    f = Featurizer(
        SOAP,
        max_size=5,
        species_list=["Fe", "O", "H"],
        kwargs={"rcut": 12, "nmax": 8, "lmax": 8},
    )
    with tempfile.TemporaryDirectory() as _tmp_dir:
        f.write_json_to_disk(write_location=_tmp_dir, json_name="test_feat.json")
        json_path = os.path.join(_tmp_dir, "test_feat.json")
        written_f = Featurizer.from_json(json_path)
        assert written_f == f


def test_featurizer_from_jsonified_dict():
    # Test generating Featurizer from a dict

    with pytest.raises(FeaturizerError):
        # catches null case
        j_dict = {}
        f = Featurizer.from_jsonified_dict(j_dict)

    # test providing only featurizer class
    j_dict = {"featurizer_class": ["dscribe.descriptors.sinematrix", "SineMatrix"]}
    f = Featurizer.from_jsonified_dict(j_dict)
    assert isinstance(f.featurization_object, SineMatrix)

    # test providing preset
    j_dict = {
        "featurizer_class": [
            "matminer.featurizers.composition.composite",
            "ElementProperty",
        ],
        "preset": "matminer",
    }
    f = Featurizer.from_jsonified_dict(j_dict)
    assert isinstance(f.featurization_object, ElementProperty)
    assert f.preset == "matminer"

    # test providing kwargs
    j_dict = {
        "featurizer_class": ["dscribe.descriptors.soap", "SOAP"],
        "kwargs": {"rcut": 6.0, "lmax": 6, "nmax": 6},
    }
    f = Featurizer.from_jsonified_dict(j_dict)
    assert isinstance(f.featurization_object, SOAP)
    assert np.isclose(f.kwargs.get("rcut"), 6.0)
    assert f.kwargs.get("lmax") == 6
    assert f.kwargs.get("nmax") == 6

    # test providing design space structures
    atoms_list = [Atoms("H"), Atoms("N")]
    encoded_atoms = [atoms_encoder(a) for a in atoms_list]
    j_dict = {
        "featurizer_class": ["dscribe.descriptors.sinematrix", "SineMatrix"],
        "design_space_structures": encoded_atoms,
    }
    f = Featurizer.from_jsonified_dict(j_dict)
    assert f.design_space_structures == atoms_list

    with pytest.raises(FeaturizerError):
        # catches that Atoms objects should be json encoded
        j_dict = {
            "featurizer_class": ["dscribe.descriptors.sinematrix", "SineMatrix"],
            "design_space_structures": atoms_list,
        }
        f = Featurizer.from_jsonified_dict(j_dict)

    with pytest.raises(FeaturizerError):
        # catches that design space structures aren't some nonsensical type
        j_dict = {
            "featurizer_class": ["dscribe.descriptors.sinematrix", "SineMatrix"],
            "design_space_structures": ["H", "N"],
        }
        f = Featurizer.from_jsonified_dict(j_dict)

    with pytest.raises(FeaturizerError):
        # catches that featurizer class must be provided
        j_dict = {"preset": "magpie"}
        f = Featurizer.from_jsonified_dict(j_dict)

    with pytest.raises(FeaturizerError):
        # catches that featurizer class must be specified correctly
        j_dict = {"featurizer_class": [SineMatrix]}
        f = Featurizer.from_jsonified_dict(j_dict)


def test_featurizer_to_jsonified_dict():
    # Tests converting a Featurizer to a dict
    surfs = flatten_structures_dict(generate_surface_structures(["Fe", "V"]))
    surfs.extend(
        flatten_structures_dict(
            generate_surface_structures(["Au", "Ag"], supercell_dim=(1, 1, 5))
        )
    )
    f = Featurizer(SineMatrix, design_space_structures=surfs, kwargs={"sparse": True})
    json_dict = f.to_jsonified_dict()
    assert json_dict["featurizer_class"] == [
        "dscribe.descriptors.sinematrix",
        "SineMatrix",
    ]
    assert json_dict["kwargs"] == {"sparse": True}

    f = Featurizer(
        SOAP,
        max_size=5,
        species_list=["O", "H", "Fe"],
        kwargs={"rcut": 12, "nmax": 8, "lmax": 8},
    )
    json_dict = f.to_jsonified_dict()
    assert json_dict["max_size"] == 5
    assert json_dict["species_list"] == ["Fe", "O", "H"]
