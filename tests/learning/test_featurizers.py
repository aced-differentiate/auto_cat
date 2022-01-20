"""Unit tests for the `autocat.learning.featurizersi` module."""

import numpy as np

from dscribe.descriptors import SineMatrix
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import ACSF
from dscribe.descriptors import SOAP

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import ChemicalSRO
from matminer.featurizers.site import OPSiteFingerprint
from matminer.featurizers.site import CrystalNNFingerprint

from autocat.adsorption import place_adsorbate
from autocat.surface import generate_surface_structures
from autocat.saa import generate_saa_structures
from autocat.learning.featurizers import Featurizer
from autocat.utils import extract_structures

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN


def test_featurizer_species_list():
    # test default species list
    f = Featurizer(SineMatrix)
    assert f.species_list == ["Fe", "Ni", "Pt", "Pd", "Cu", "C", "N", "O", "H"]

    # test updating species list manually and sorting
    f.species_list = ["Li", "Na", "K"]
    assert f.species_list == ["K", "Na", "Li"]

    # test getting species list from design space structures
    surfs = extract_structures(generate_surface_structures(["Fe", "V", "Ti"]))
    saas = extract_structures(generate_saa_structures(["Cu", "Au"], ["Fe", "Pt"]))
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
    surfs = extract_structures(
        generate_surface_structures(["Ru"], supercell_dim=(2, 2, 4))
    )
    surfs.extend(
        extract_structures(generate_surface_structures(["Fe"], supercell_dim=(4, 4, 4)))
    )
    f.design_space_structures = surfs
    assert f.max_size == 64


def test_featurizer_design_space_structures():
    # tests giving design space structures
    surfs = extract_structures(generate_surface_structures(["Li", "Na"]))
    surfs.extend(
        extract_structures(
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
    saa = extract_structures(generate_saa_structures(["Cu"], ["Pt"]))[0]
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
    ads_struct = extract_structures(place_adsorbate(saa, "OH", position=(0.0, 0.0)))[0]
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
    saas = extract_structures(
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
            extract_structures(place_adsorbate(struct, "NNH", position=(0.0, 0.0)))[0]
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
