"""Unit tests for the `autocat.utils` module"""

from ase import Atoms
from ase.build import fcc100
from ase.build import fcc111
from ase.build import bcc110

from autocat.surface import generate_surface_structures
from autocat.saa import generate_saa_structures
from autocat.adsorption import generate_adsorbed_structures
from autocat.utils import flatten_structures_dict


def test_extract_surfaces():
    # Tests extracting structures from `autocat.surface.generate_surface_structures`
    surfaces = generate_surface_structures(
        ["Pt", "Cu", "Li"], facets={"Pt": ["100", "111"], "Cu": ["111"], "Li": ["110"]}
    )
    ex_structures = flatten_structures_dict(surfaces)
    assert all(isinstance(struct, Atoms) for struct in ex_structures)
    # checks atoms objects left untouched during extraction
    pt_struct100 = fcc100("Pt", size=(3, 3, 4), vacuum=10)
    assert pt_struct100 in ex_structures
    pt_struct111 = fcc111("Pt", size=(3, 3, 4), vacuum=10)
    assert pt_struct111 in ex_structures
    cu_struct = fcc111("Cu", size=(3, 3, 4), vacuum=10)
    assert cu_struct in ex_structures
    li_struct = bcc110("Li", size=(3, 3, 4), vacuum=10)
    assert li_struct in ex_structures


def test_extract_saas():
    # Tests extracting saa structures
    saas = generate_saa_structures(
        ["Cu", "Au"],
        ["Fe"],
        facets={"Cu": ["110"], "Au": ["100"]},
        supercell_dim=(2, 2, 5),
    )
    ex_structures = flatten_structures_dict(saas)
    assert all(isinstance(struct, Atoms) for struct in ex_structures)
    assert saas["Cu"]["Fe"]["fcc110"]["structure"] in ex_structures
    assert saas["Au"]["Fe"]["fcc100"]["structure"] in ex_structures


def test_extract_adsorption():
    # Test extracting adsorption structures
    saa = generate_saa_structures(["Ru"], ["Pd"], supercell_dim=(2, 2, 5),)["Ru"]["Pd"][
        "hcp0001"
    ]["structure"]
    ads_dict = generate_adsorbed_structures(
        saa,
        adsorbates=["NH2", "Li"],
        adsorption_sites={"custom": [(0.0, 0.0)]},
        use_all_sites=False,
    )
    ex_structures = flatten_structures_dict(ads_dict)
    assert all(isinstance(struct, Atoms) for struct in ex_structures)
    assert ads_dict["NH2"]["custom"]["0.0_0.0"]["structure"] in ex_structures
    assert ads_dict["Li"]["custom"]["0.0_0.0"]["structure"] in ex_structures
