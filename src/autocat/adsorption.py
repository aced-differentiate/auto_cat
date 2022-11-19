import os
import numpy as np
import itertools
from typing import List
from typing import Tuple
from typing import Dict
from typing import Union
from typing import Any
from typing import Sequence

from ase import Atoms
from ase.build import add_adsorbate
from ase.build import molecule as build_molecule
from ase.data import chemical_symbols
from ase.data import atomic_numbers
from ase.data import covalent_radii
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
from pymatgen.analysis.structure_matcher import StructureMatcher

from autocat.data.intermediates import NRR_MOLS
from autocat.data.intermediates import ORR_MOLS


# custom types for readability
RotationOperation = Sequence[Union[float, str]]
RotationOperations = Sequence[RotationOperation]
AdsorptionSite = Dict[str, Sequence[Sequence[float]]]


class AutocatAdsorptionGenerationError(Exception):
    pass


def generate_molecule(
    molecule_name: str = None,
    rotations: RotationOperations = None,
    cell: Sequence[float] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Generates an `ase.Atoms` object of an isolated molecule.
    If specified, can write out a .traj file containing the isolated molecule
    in a box.

    Parameters
    ----------

    molecule_name (REQUIRED):
        String of the name of the molecule to be generated. Will search in
        `autocat.intermediates` data first and then `ase` g2 database.

    rotations:
        List of rotation operations to be applied to the adsorbate
        molecule/intermediate before being placed on the host surface.

        Example:
        Rotating 90 degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [(90.0, "z"), (45.0, "y")]

        Defaults to [(0, "x")] (i.e., no rotations applied).

    cell:
        List of float specifying the dimensions of the box to place the molecule
        in, in Angstrom.
        Defaults to [15, 15, 15].

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the molecule structure must be written.
        The molecule is written to disk at
        [write_location]/references/[molecule_name]/input.traj

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    Dictionary containing Atoms object of the generated molecule and path to
    .traj file if written to disk.

    Example:
    {
        "NH": {"structure": NH_ase_obj, "traj_file_path": "/path/to/NH/traj/file"},
    }
    """
    if molecule_name is None:
        msg = "Molecule name must be specified"
        raise AutocatAdsorptionGenerationError(msg)

    if rotations is None:
        rotations = [[0.0, "x"]]

    if cell is None:
        cell = [15, 15, 15]

    m = None
    if molecule_name in NRR_MOLS:
        m = NRR_MOLS[molecule_name].copy()
    elif molecule_name in ORR_MOLS:
        m = ORR_MOLS[molecule_name].copy()
    elif molecule_name in chemical_symbols:  # atom-in-a-box
        m = Atoms(molecule_name)
    elif molecule_name in g2.names:
        m = build_molecule(molecule_name)

    if m is None:
        msg = f"Unable to construct molecule {molecule_name}"
        raise NotImplementedError(msg)

    for r in rotations:
        m.rotate(r[0], r[1])

    m.cell = cell

    m.center()

    traj_file_path = None
    if write_to_disk:
        dir_path = os.path.join(write_location, "references", f"{molecule_name}")
        os.makedirs(dir_path, exist_ok=dirs_exist_ok)
        traj_file_path = os.path.join(dir_path, "input.traj")
        m.write(traj_file_path)
        print(f"{molecule_name} molecule structure written to {traj_file_path}")

    return {"structure": m, "traj_file_path": traj_file_path}


def adsorption_sites_to_possible_ads_site_list(
    adsorption_sites: Union[Dict[str, AdsorptionSite], AdsorptionSite] = None,
    adsorbates: Union[Dict[str, Union[str, Atoms]], Sequence[str]] = None,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Takes adsorption sites and converts to a list where each index
    corresponds to a specific list and is a list of possible adsorbates
    at that site.

    Parameters
    ----------

    adsorption_sites (REQUIRED):
        List of xy coordinates of sites on the surface
        where any of the adsorbates can be placed.
        Alternatively, a single dictionary with label and a list of xy
        coordinates can be provided as input to be used to indicate a separate
        list of potential sites for each adsorbate.

        Example:

        [(0.0, 0.0), (0.25, 0.25), (0.75, 0.25), (0.5, 0.5)]

        OR

        {
            "OH": [(0.0, 0.0), (0.25, 0.25)]
            "H2O": [(0.5, 0.5), (0.75, 0.25)]
        }

    adsorbates:
        Dictionary of adsorbate molecule/intermediate names and corresponding
        `ase.Atoms` object or string to be placed on the host surface.

        Note that the strings that appear as values must be in the list of
        supported molecules in `autocat.data.intermediates` or in the `ase` g2
        database. Predefined data in `autocat.data` will take priority over that
        in `ase`.

        Alternatively, a list of strings can be provided as input.
        Note that each string has to be *unique* and available in
        `autocat.data.intermediates` or the `ase` g2 database.

        Example:
        {
            "NH": "NH",
            "N*": "N",
            "NNH": NNH_atoms_obj,
            ...
        }

        OR

        ["NH", "NNH"]

        Required if providing `adsorption_sites` as a list

    Returns
    -------

    List where each index is the possible adsorbates at each site
    and a list of all sites

    Example:

    [["OH", "H"], ["H"], ["OH", "OOH"]]

    AND

    [[0.0, 0.0], [0.25, 0.25], [0.7, 0.6]]

    """
    if adsorption_sites is None:
        msg = "Adsorption sites must be provided"
        raise AutocatAdsorptionGenerationError(msg)

    if isinstance(adsorbates, dict):
        adsorbates = list(adsorbates.keys())

    if isinstance(adsorption_sites, list):
        if adsorbates is None:
            msg = "Adsorbates must be provided if adsorption sites given as a list"
            raise AutocatAdsorptionGenerationError(msg)
        elif len(adsorption_sites) > len(np.unique(adsorption_sites, axis=0)):
            adsorption_sites = [
                tuple(row) for row in np.unique(adsorption_sites, axis=0)
            ]
        possible_ads_site_list = [adsorbates for i in range(len(adsorption_sites))]
        sites = adsorption_sites

    elif isinstance(adsorption_sites, dict):
        sites = []
        possible_ads_site_list = []
        for ads_sym, ads_sites in adsorption_sites.items():
            for ads_site in ads_sites:
                if ads_site not in sites:
                    sites.append(ads_site)
                    possible_ads_site_list.append([ads_sym])
                else:
                    idx = sites.index(ads_site)
                    possible_ads_site_list[idx].extend([ads_sym])

    return possible_ads_site_list, sites


def enumerate_adsorbed_site_list(
    adsorption_sites: Union[Dict[str, AdsorptionSite], AdsorptionSite] = None,
    adsorbates: Union[Dict[str, Union[str, Atoms]], Sequence[str]] = None,
    adsorbate_coverage: Dict[str, Union[float, int]] = None,
) -> Tuple[List[List[str]], List[Tuple[float]]]:
    """
    Generates all adsorption combinations which are restricted by coverages.

    Parameters
    ----------

    adsorption_sites (REQUIRED):
        List of xy coordinates of sites on the surface
        where any of the adsorbates can be placed.
        Alternatively, a single dictionary with label and a list of xy
        coordinates can be provided as input to be used to indicate a separate
        list of potential sites for each adsorbate.

        Example:

        [(0.0, 0.0), (0.25, 0.25), (0.75, 0.25), (0.5, 0.5)]

        OR

        {
            "OH": [(0.0, 0.0), (0.25, 0.25)]
            "H2O": [(0.5, 0.5), (0.75, 0.25)]
        }

    adsorbates (REQUIRED):
        Dictionary of adsorbate molecule/intermediate names and corresponding
        `ase.Atoms` object or string to be placed on the host surface.

        Note that the strings that appear as values must be in the list of
        supported molecules in `autocat.data.intermediates` or in the `ase` g2
        database. Predefined data in `autocat.data` will take priority over that
        in `ase`.

        Alternatively, a list of strings can be provided as input.
        Note that each string has to be *unique* and available in
        `autocat.data.intermediates` or the `ase` g2 database.

        Example:
        {
            "NH": "NH",
            "N*": "N",
            "NNH": NNH_atoms_obj,
            ...
        }

        OR

        ["NH", "NNH"]

        Required if providing `adsorption_sites` as a list

    adsorbate_coverage (REQUIRED):
        Dictionary indicating the desired coverage of each `adsorbate` specified
        in `adsorbates`. If each value is an int and >= 1, then it is assumed that
        this is the maximum number of each adsorbate to place. Otherwise, these values are
        interpreted as percentages of the number of sites. If you would like a specific
        adsorbate to not have any coverage constraints, set it to `np.inf`

        Example:
        {
            "OH": 2,
            "CO": 3,
        }

        OR

        {
            "OH": 0.25,
            "CO": 0.75
        }

    Returns
    -------

        List of all enumerated combinations of adsorbates placed subject to the maximum
        concentration constraint and corresponding list of sites

        Example:

        [
            ["OH", "OH", "CO"],
            ["OH", "CO", "OH"],
            ["CO", "OH", "OH"]
        ]

        AND

        [(0.0), (0.25, 0.25), (0.4, 0.6)]
    """
    if adsorbates is None:
        msg = "Adsorbates must be provided"
        raise AutocatAdsorptionGenerationError(msg)

    if adsorbate_coverage is None:
        msg = "Adsorbate coverage must be provided"
        raise AutocatAdsorptionGenerationError(msg)

    for ads in adsorbates:
        if ads not in adsorbate_coverage:
            msg = f"Coverage not specified for {ads}"
            raise AutocatAdsorptionGenerationError(msg)

    nested_ads_combos, sites = adsorption_sites_to_possible_ads_site_list(
        adsorption_sites=adsorption_sites, adsorbates=adsorbates
    )

    # convert cov percent to max num of each adsorbate
    if not (
        all((isinstance(val, int) and val >= 1) for val in adsorbate_coverage.values())
    ):
        total_num_sites = len(sites)
        ads_cov = {}
        for ads, cov in adsorbate_coverage.items():
            if np.isinf(cov):
                ads_cov[ads] = np.inf
            else:
                ads_cov[ads] = int(np.floor(cov * total_num_sites))
        adsorbate_coverage = ads_cov

    all_ads_combos = itertools.product(*nested_ads_combos)

    filtered_ads_combos = []
    for ads_combo in all_ads_combos:
        over_cov_limit = False
        ads_combo = list(ads_combo)
        for ads in ads_combo:
            if ads_combo.count(ads) > adsorbate_coverage[ads]:
                over_cov_limit = True
                break
        if not over_cov_limit:
            filtered_ads_combos.append(ads_combo)

    # check that combinations given constraints were found
    if not filtered_ads_combos:
        msg = """
        Unable to enumerate adsorption sites.
        This is most likely due to too restrictive maximum coverages.
        Please consider allowing unoccupied sites by using `X` in
        `adsorbates`, `adsorbate_coverage`, and `adsorption_sites`
        """
        raise AutocatAdsorptionGenerationError(msg)

    return filtered_ads_combos, sites


def place_multiple_adsorbates(
    surface: Atoms = None,
    adsorbates: Union[Dict[str, Union[str, Atoms]], Sequence[str]] = None,
    adsorbates_at_each_site: Sequence[str] = None,
    adsorption_sites_list: Sequence[Sequence[float]] = None,
    heights: Union[Dict[str, float], float] = None,
    anchor_atom_indices: Union[Dict[str, int], int] = None,
    rotations: Union[Dict[str, RotationOperations], RotationOperations] = None,
) -> Atoms:
    """
    Given a list of adsorbates for each desired site and a corresponding
    site list, place the adsorbates at each site

    Parameters
    ----------

    surface (REQUIRED):
        Atoms object for the structure of the host surface.

    adsorbates (REQUIRED):
        Dictionary of adsorbate molecule/intermediate names and corresponding
        `ase.Atoms` object or string to be placed on the host surface.

        Note that the strings that appear as values must be in the list of
        supported molecules in `autocat.data.intermediates` or in the `ase` g2
        database. Predefined data in `autocat.data` will take priority over that
        in `ase`.

        Alternatively, a list of strings can be provided as input.
        Note that each string has to be *unique* and available in
        `autocat.data.intermediates` or the `ase` g2 database.

        Example:
        {
            "NH": "NH",
            "N*": "N",
            "NNH": NNH_atoms_obj,
            ...
        }

        OR

        ["NH", "NNH"]

    adsorbates_at_each_site (REQUIRED):
        List of adsorbates to be placed at each site specified in `sites_list`.
        Must be the same length as `sites_list` with all adsorbates
        specified in `adsorbates`.

        Example:
        ["OH", "H", "OH"]

    sites_list (REQUIRED):
        List of xy-coords of each site corresponding to the list
        given by `adsorbates_at_each_site`

        Example:
        [(0.0, 0.0), (0.25, 0.25), (0.7, 0.6)]

    rotations:
        Dictionary of the list of rotation operations to be applied to each
        adsorbate molecule/intermediate type before being placed on the host surface.
        Alternatively, a single list of rotation operations can be provided as
        input to be used for all adsorbates.

        Rotating 90 degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [(90.0, "z"), (45.0, "y")]

        Example:
        {
            "NH": [(90.0, "z"), (45.0, "y")],
            "NNH": ...
        }

        Defaults to [(0, "x")] (i.e., no rotations applied) for each adsorbate
        molecule.

    heights:
        Dictionary of the height above surface where each adsorbate type should be
        placed.
        Alternatively, a single float value can be provided as input to be
        used for all adsorbates.
        If None, will estimate initial height based on covalent radii of the
        nearest neighbor atoms for each adsorbate.

    anchor_atom_indices:
        Dictionary of the integer index of the atom in each adsorbate molecule
        that should be used as anchor when placing it on the surface.
        Alternatively, a single integer index can be provided as input to be
        used for all adsorbates.
        Defaults to the atom at index 0 for each adsorbate molecule.

    Returns
    -------

    Atoms object of surface structure with adsorbates placed at specified sites

    """
    # input wrangling
    if surface is None:
        msg = "Surface must be provided"
        raise AutocatAdsorptionGenerationError(msg)

    if adsorbates is None:
        msg = "Adsorbates must be provided"
        raise AutocatAdsorptionGenerationError(msg)

    if adsorption_sites_list is None:
        msg = "List of sites must be provided"
        raise AutocatAdsorptionGenerationError(msg)
    elif len(adsorption_sites_list) > len(np.unique(adsorption_sites_list, axis=0)):
        msg = "Cannot place multiple adsorbates simultaneously at the same site"
        raise AutocatAdsorptionGenerationError(msg)

    if adsorbates_at_each_site is None:
        msg = "List of adsorbates to place at each site must be provided"
        raise AutocatAdsorptionGenerationError(msg)
    elif not isinstance(adsorbates_at_each_site, list):
        msg = f"Unsupported type for adsorbates_at_each_site {type(adsorbates_at_each_site)}"
        raise AutocatAdsorptionGenerationError(msg)
    elif not all(isinstance(ads, str) for ads in adsorbates_at_each_site):
        msg = "List of adsorbates to place at each site must be flat and contain only strings"
        raise AutocatAdsorptionGenerationError(msg)
    elif len(adsorbates_at_each_site) != len(adsorption_sites_list):
        msg = "List of adsorbates to be placed must be same length as sites list"
        raise AutocatAdsorptionGenerationError(msg)

    # get Atoms objects for each adsorbate molecule
    if isinstance(adsorbates, dict):
        ads_mols = []
        for ads in adsorbates_at_each_site:
            if isinstance(adsorbates[ads], str):
                mol = generate_molecule(adsorbates[ads]).get("structure")
            elif isinstance(adsorbates[ads], Atoms):
                mol = adsorbates[ads]
            else:
                msg = f"Unrecognized format for adsorbate {ads}"
                raise AutocatAdsorptionGenerationError(msg)
            ads_mols.append(mol)
    elif isinstance(adsorbates, list):
        ads_mols = [
            generate_molecule(ads_str).get("structure")
            for ads_str in adsorbates_at_each_site
        ]
    else:
        msg = f"Unrecognized format for `adsorbates` {type(adsorbates)}"
        raise AutocatAdsorptionGenerationError(msg)

    # get lists of height, anchor atoms, and rotations for each site
    if heights is None:
        heights_list = [None] * len(adsorption_sites_list)
    elif isinstance(heights, dict):
        heights_list = [heights.get(ads, None) for ads in adsorbates_at_each_site]
    elif isinstance(heights, float):
        heights_list = [heights] * len(adsorption_sites_list)

    if anchor_atom_indices is None:
        anchor_list = [0] * len(adsorption_sites_list)
    elif isinstance(anchor_atom_indices, dict):
        anchor_list = [
            anchor_atom_indices.get(ads, 0) for ads in adsorbates_at_each_site
        ]
    elif isinstance(anchor_atom_indices, int):
        anchor_list = [anchor_atom_indices] * len(adsorption_sites_list)

    if rotations is None:
        rots_list = [None] * len(adsorption_sites_list)
    elif isinstance(rotations, dict):
        rots_list = [rotations.get(ads, None) for ads in adsorbates_at_each_site]
    elif isinstance(rotations, (list, tuple)):
        rots_list = [rotations] * len(adsorption_sites_list)

    ads_surface = surface.copy()
    for mol, site, height, anchor_idx, rotation in zip(
        ads_mols, adsorption_sites_list, heights_list, anchor_list, rots_list
    ):
        ads_surface = place_adsorbate(
            surface=ads_surface,
            adsorbate=mol,
            adsorption_site=site,
            height=height,
            rotations=rotation,
            anchor_atom_index=anchor_idx,
        )
    return ads_surface


def generate_high_coverage_adsorbed_structures(
    surface: Union[str, Atoms] = None,
    adsorbates: Union[Dict[str, Union[str, Atoms]], Sequence[str]] = None,
    adsorbate_coverage: Dict[str, Union[float, int]] = None,
    adsorption_sites: Union[Dict[str, AdsorptionSite], AdsorptionSite] = None,
    use_all_sites: bool = None,
    site_types: Union[str, Sequence[str], Dict[str, Sequence[str]]] = None,
    heights: Union[Dict[str, float], float] = None,
    anchor_atom_indices: Union[Dict[str, int], int] = None,
    rotations: Union[Dict[str, RotationOperations], RotationOperations] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """

    Builds structures with multiple adsorbates for high coverage systems.
    For a given composition of adsorbates, all unique structures will be
    enumerated and returned.

    Parameters
    ----------

    surface (REQUIRED):
        Atoms object or path to a file on disk containing the structure of the
        host surface.
        Note that the format of the file must be readable with `ase.io`.

    adsorbates (REQUIRED):
        Dictionary of adsorbate molecule/intermediate names and corresponding
        `ase.Atoms` object or string to be placed on the host surface.

        Note that the strings that appear as values must be in the list of
        supported molecules in `autocat.data.intermediates` or in the `ase` g2
        database. Predefined data in `autocat.data` will take priority over that
        in `ase`.

        Alternatively, a list of strings can be provided as input.
        Note that each string has to be *unique* and available in
        `autocat.data.intermediates` or the `ase` g2 database.

        Example:
        {
            "NH": "NH",
            "N*": "N",
            "NNH": NNH_atoms_obj,
            ...
        }

        OR

        ["NH", "NNH"]

    adsorbate_coverage (REQUIRED):
        Dictionary indicating the desired coverage of each `adsorbate` specified
        in `adsorbates`. If each value is an int and >= 1, then it is assumed that
        this is the maximum number of each adsorbate to place. Otherwise, these values are
        interpreted as percentages of the number of sites. If you would like a specific
        adsorbate to not have any coverage constraints, set it to `np.inf`

        Example:
        {
            "OH": 2,
            "CO": 3,
        }

        OR

        {
            "OH": 0.25,
            "CO": 0.75
        }

    rotations:
        Dictionary of the list of rotation operations to be applied to each
        adsorbate molecule/intermediate before being placed on the host surface.
        Alternatively, a single list of rotation operations can be provided as
        input to be used for all adsorbates.

        Rotating 90 degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [(90.0, "z"), (45.0, "y")]

        Example:
        {
            "NH": [(90.0, "z"), (45.0, "y")],
            "NNH": ...
        }

        Defaults to [(0, "x")] (i.e., no rotations applied) for each adsorbate
        molecule.

    adsorption_sites:
        List of xy coordinates of sites on the surface
        where any of the adsorbates can be placed.
        Alternatively, a single dictionary with label and a list of xy
        coordinates can be provided as input to be used to indicate a separate
        list of potential sites for each adsorbate.

        Example:

        [(0.0, 0.0), (0.25, 0.25), (0.75, 0.25), (0.5, 0.5)]

        OR

        {
            "OH": [(0.0, 0.0), (0.25, 0.25)]
            "H2O": [(0.5, 0.5), (0.75, 0.25)]
        }

    use_all_sites:
        Bool specifying if the adsorption sites should be taken to be all of the
        unique sites of type `site_type`

        NB: If True, overrides any sites defined in `adsorption_sites` for each
        adsorbate.

        Defaults to True.

    site_types:
        Indicates which types of surface site should be occupied by the
        adsorbates. Can be a string or list which is applied to all adsorbates,
        or a dict indicating site types for each adsorbate.
        Ignored if `use_all_sites` is False.

        Examples:

        "ontop"

        OR

        ["ontop", "hollow"]

        OR

        {"OH": ["ontop", "hollow"], "H": ["hollow"]}

        Defaults to ["ontop", "bridge", "hollow"] (if `use_all_sites` is True)
        for all adsorbates.

    heights:
        Dictionary of the height above surface where each adsorbate should be
        placed.
        Alternatively, a single float value can be provided as input to be
        used for all adsorbates.
        If None, will estimate initial height based on covalent radii of the
        nearest neighbor atoms for each adsorbate.

    anchor_atom_indices:
        Dictionary of the integer index of the atom in each adsorbate molecule
        that should be used as anchor when placing it on the surface.
        Alternatively, a single integer index can be provided as input to be
        used for all adsorbates.
        Defaults to the atom at index 0 for each adsorbate molecule.

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.
        In the specified write_location, the following directory structure
        will be created:

        adsorbates/[adsorbate_1]/[site_label_1]/[xy_site_coord_1]/input.traj
        adsorbates/[adsorbate_1]/[site_label_1]/[xy_site_coord_2]/input.traj
        ...
        adsorbates/[adsorbate_1]/[site_label_2]/[xy_site_coord_1]/input.traj
        ...
        adsorbates/[adsorbate_2]/[site_label_1]/[xy_site_coord_1]/input.traj
        ...

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    ads_structures:
        Dictionary containing all of the configurations of placing the
        multiple adsorbates

        Example:

        {
            1: {
                "structure": structure1,
                "traj_file_path": "/path/to/structure1/traj/file"
            },
            2: {
                "structure": structure2,
                "traj_file_path": "/path/to/structure2/traj/file"
            }
            ...
        }

    """
    if surface is None:
        msg = "Surface structure must be specified"
        raise AutocatAdsorptionGenerationError(msg)
    elif not isinstance(surface, Atoms):
        msg = f"Unrecognized input type for surface ({type(surface)})"
        raise AutocatAdsorptionGenerationError(msg)

    # Input wrangling for the different types of parameter values allowed for
    # the same function arguments.

    if adsorbates is None or not adsorbates:
        msg = "Adsorbate molecules/intermediates must be specified"
        raise AutocatAdsorptionGenerationError(msg)
    elif isinstance(adsorbates, (list, tuple)):
        adsorbates = {ads_key: ads_key for ads_key in adsorbates}
    elif not isinstance(adsorbates, dict):
        msg = f"Unrecognized input type for adsorbates ({type(adsorbates)})"
        raise AutocatAdsorptionGenerationError(msg)

    if adsorbate_coverage is None or not adsorbate_coverage:
        msg = "Adsorbate molecules/intermediates must be specified"
        raise AutocatAdsorptionGenerationError(msg)
    elif not isinstance(adsorbate_coverage, dict):
        msg = f"Unrecognized input type for adsorbate_coverage ({type(adsorbate_coverage)})"
        raise AutocatAdsorptionGenerationError(msg)

    if rotations is None:
        rotations = {}
    elif isinstance(rotations, (list, tuple)):
        rotations = {ads_key: rotations for ads_key in adsorbates}
    elif not isinstance(rotations, dict):
        msg = f"Unrecognized input type for rotations ({type(rotations)})"
        raise AutocatAdsorptionGenerationError(msg)

    if adsorption_sites is None:
        adsorption_sites = {}
    elif not isinstance(adsorption_sites, (dict, list)):
        msg = f"Unrecognized input type for adsorption_sites ({type(adsorption_sites)})"
        raise AutocatAdsorptionGenerationError(msg)

    if use_all_sites is None:
        use_all_sites = False
    elif not isinstance(use_all_sites, bool):
        msg = f"Unrecognized input type for use_all_sites ({type(use_all_sites)})"
        raise AutocatAdsorptionGenerationError(msg)

    if site_types is None:
        site_types = {ads_key: ["ontop", "bridge", "hollow"] for ads_key in adsorbates}
    elif isinstance(site_types, dict):
        for val in site_types.values():
            if not isinstance(val, list):
                msg = "All site types in dict must be given as a list for each value"
                raise AutocatAdsorptionGenerationError(msg)
            elif False in [s in ["ontop", "bridge", "hollow"] for s in val]:
                msg = f"Unrecognized site type in {val}"
                raise AutocatAdsorptionGenerationError(msg)
    elif isinstance(site_types, list):
        if False in [s in ["ontop", "bridge", "hollow"] for s in site_types]:
            msg = f"Unrecognized site type in {site_types}"
            raise AutocatAdsorptionGenerationError(msg)
        site_types = {ads_key: site_types for ads_key in adsorbates}
    elif isinstance(site_types, str):
        if site_types not in ["ontop", "hollow", "bridge"]:
            msg = f"Unrecognized site type {site_types}"
            raise AutocatAdsorptionGenerationError(msg)
        site_types = {ads_key: [site_types] for ads_key in adsorbates}
    elif not isinstance(site_types, str):
        msg = f"Unrecognized input type for site_type ({type(site_types)})"
        raise AutocatAdsorptionGenerationError(msg)

    if heights is None:
        heights = {}
    elif isinstance(heights, float):
        heights = {ads_key: heights for ads_key in adsorbates}
    elif not isinstance(heights, dict):
        msg = f"Unrecognized input type for heights ({type(heights)})"
        raise AutocatAdsorptionGenerationError(msg)

    if anchor_atom_indices is None:
        anchor_atom_indices = {}
    elif isinstance(anchor_atom_indices, int):
        anchor_atom_indices = {ads_key: anchor_atom_indices for ads_key in adsorbates}
    elif not isinstance(anchor_atom_indices, dict):
        msg = f"Unrecognized input type for anchor_atom_indices ({type(anchor_atom_indices)})"
        raise AutocatAdsorptionGenerationError(msg)

    # get all adsorption sites if needed
    if use_all_sites:
        adsorption_sites = {}
        for ads_key in site_types:
            sites_dict = get_adsorption_sites(
                surface, site_types=site_types[ads_key], symm_reduce=False
            )
            sites = []
            for st in sites_dict:
                sites.extend(sites_dict[st])
            xy_sites = [tuple(s[:2]) for s in sites]
            adsorption_sites[ads_key] = xy_sites

    # check that all adsorbates are specified in adsorbate_coverage
    for ads_key in adsorbates:
        if ads_key not in adsorbate_coverage:
            msg = f"{ads_key} not specified in adsorbate_coverage"
            raise AutocatAdsorptionGenerationError(msg)

    # check that all adsorbates in adsorbate_coverage are present in adsorbates
    for ads_key in adsorbate_coverage:
        if ads_key not in adsorbates:
            msg = f"{ads_key} specified in adsorbate_coverage but not adsorbates"
            raise AutocatAdsorptionGenerationError(msg)

    # enumerate all possible adsorbate placement combinations
    enum_ads_at_each_site, sites_list = enumerate_adsorbed_site_list(
        adsorption_sites=adsorption_sites,
        adsorbates=adsorbates,
        adsorbate_coverage=adsorbate_coverage,
    )

    # rm trivial case of all vacancies
    try:
        enum_ads_at_each_site.remove(["X"] * len(sites_list))
    except ValueError:
        pass

    # generate each combination of adsorbate placement on the surface
    adsorbed_struct_list = []
    for ads_combo in enum_ads_at_each_site:
        adsorbed_structure = place_multiple_adsorbates(
            surface=surface,
            adsorbates=adsorbates,
            adsorbates_at_each_site=ads_combo,
            adsorption_sites_list=sites_list,
            heights=heights,
            anchor_atom_indices=anchor_atom_indices,
            rotations=rotations,
        )
        adsorbed_struct_list.append(adsorbed_structure)

    # rm X adsorbates
    for struct in adsorbed_struct_list:
        idx_to_delete = []
        for idx, atom in enumerate(struct):
            if atom.symbol == "X":
                idx_to_delete.append(idx)
        idx_to_delete.sort(reverse=True)
        for idx in idx_to_delete:
            del struct[idx]

    # filter structures by matching symmetries
    matcher = StructureMatcher()
    conv = AseAtomsAdaptor()
    adsorbed_struct_list_pym = [
        conv.get_structure(ase_struct) for ase_struct in adsorbed_struct_list
    ]
    filtered_adsorbed_struct_list_pym = [
        struct_group[0]
        for struct_group in matcher.group_structures(adsorbed_struct_list_pym)
    ]
    filtered_adsorbed_struct_list = [
        conv.get_atoms(pym_struct) for pym_struct in filtered_adsorbed_struct_list_pym
    ]

    ads_dict = {}
    for idx, filt_ads_struct in enumerate(filtered_adsorbed_struct_list):
        traj_file_path = None
        if write_to_disk:
            dir_path = os.path.join(write_location, "multiple_adsorbates", str(idx),)
            os.makedirs(dir_path, exist_ok=dirs_exist_ok)
            traj_file_path = os.path.join(dir_path, "input.traj")
            filt_ads_struct.write(traj_file_path)
            print(f"Adsorbate combination #{idx} written to {traj_file_path}")
        ads_dict[idx] = {"structure": filt_ads_struct, "traj_file_path": traj_file_path}

    return ads_dict


def generate_adsorbed_structures(
    surface: Union[str, Atoms] = None,
    adsorbates: Union[Dict[str, Union[str, Atoms]], Sequence[str]] = None,
    adsorption_sites: Union[Dict[str, AdsorptionSite], AdsorptionSite] = None,
    use_all_sites: Union[Dict[str, bool], bool] = None,
    site_types: Union[Dict[str, Sequence[str]], Sequence[str]] = None,
    heights: Union[Dict[str, float], float] = None,
    anchor_atom_indices: Union[Dict[str, int], int] = None,
    rotations: Union[Dict[str, RotationOperations], RotationOperations] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """

    Builds structures for reaction intermediates by placing the adsorbate moeity
    on the input surface at specified positions.

    Parameters
    ----------

    surface (REQUIRED):
        Atoms object or path to a file on disk containing the structure of the
        host surface.
        Note that the format of the file must be readable with `ase.io`.

    adsorbates (REQUIRED):
        Dictionary of adsorbate molecule/intermediate names and corresponding
        `ase.Atoms` object or string to be placed on the host surface.

        Note that the strings that appear as values must be in the list of
        supported molecules in `autocat.data.intermediates` or in the `ase` g2
        database. Predefined data in `autocat.data` will take priority over that
        in `ase`.

        Alternatively, a list of strings can be provided as input.
        Note that each string has to be *unique* and available in
        `autocat.data.intermediates` or the `ase` g2 database.

        Example:
        {
            "NH": "NH",
            "N*": "N",
            "NNH": NNH_atoms_obj,
            ...
        }

        OR

        ["NH", "NNH"]

    rotations:
        Dictionary of the list of rotation operations to be applied to each
        adsorbate molecule/intermediate before being placed on the host surface.
        Alternatively, a single list of rotation operations can be provided as
        input to be used for all adsorbates.

        Rotating 90 degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [(90.0, "z"), (45.0, "y")]

        Example:
        {
            "NH": [(90.0, "z"), (45.0, "y")],
            "NNH": ...
        }

        Defaults to [(0, "x")] (i.e., no rotations applied) for each adsorbate
        molecule.

    adsorption_sites:
        Dictionary with labels + list of xy coordinates of sites on the surface
        where each adsorbate must be placed.
        Alternatively, a single dictionary with label and a list of xy
        coordinates can be provided as input to be used for all adsorbates.

        Example:
        {
            "NH": {
                "my_awesome_site_1": [(0.0, 0.0), (0.25, 0.25)],
                "my_awesome_site_2": [(0.0, 1.5)],
                ...
            },
            "NNH": ...
        }

        OR

        {
            "my_default_site": [(0.5, 0.5)],
        }

        Defaults to {"origin": [(0, 0)]} for all adsorbates.

    use_all_sites:
        Dictionary specifying if all symmetrically unique sites on the surface
        must be used for placing each adsorbate. Will generate a number of
        adsorbed structures equal to the number of symmetrically unique sites
        identified using the `AdsorbateSiteFinder` module from `pymatgen`.
        Alternatively, a single Boolean value can be provided as input to be
        used for all adsorbates.

        NB: If True, overrides any sites defined in `adsorption_sites` for each
        adsorbate.

        Defaults to False for all adsorbates.

    site_types:
        Dictionary of the types of adsorption sites to be searched for for each
        adsorbate. Options are "ontop", "bridge", and "hollow".
        Alternatively, a single list of adsorption sites can be provided as
        input to be used for all adsorbates.
        Ignored if `use_all_sites` is False.

        Defaults to ["ontop", "bridge", "hollow"] (if `use_all_sites` is True)
        for all adsorbates.

    heights:
        Dictionary of the height above surface where each adsorbate should be
        placed.
        Alternatively, a single float value can be provided as input to be
        used for all adsorbates.
        If None, will estimate initial height based on covalent radii of the
        nearest neighbor atoms for each adsorbate.

    anchor_atom_indices:
        Dictionary of the integer index of the atom in each adsorbate molecule
        that should be used as anchor when placing it on the surface.
        Alternatively, a single integer index can be provided as input to be
        used for all adsorbates.
        Defaults to the atom at index 0 for each adsorbate molecule.

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.
        In the specified write_location, the following directory structure
        will be created:

        adsorbates/[adsorbate_1]/[site_label_1]/[xy_site_coord_1]/input.traj
        adsorbates/[adsorbate_1]/[site_label_1]/[xy_site_coord_2]/input.traj
        ...
        adsorbates/[adsorbate_1]/[site_label_2]/[xy_site_coord_1]/input.traj
        ...
        adsorbates/[adsorbate_2]/[site_label_1]/[xy_site_coord_1]/input.traj
        ...

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    ads_structures:
        Dictionary containing all of the reaction structures (with the input
        molecule/intermediate adsorbed onto the surface) as `ase.Atoms` object
        for all adsorbates and adsorption sites.

        Example:

        {
            "NH": {
                "ontop": {
                    "5.345_2.342": {
                        "structure": NH_on_surface_obj,
                        "traj_file_path": "/path/to/NH/adsorbed/on/surface/traj/file"
                    },
                    "1.478_1.230": {
                        ...
                    },
                },
                "hollow": {
                    ...
                },
                ...
            "NNH": {
                ...
            }
        }

    """
    if surface is None:
        msg = "Surface structure must be specified"
        raise AutocatAdsorptionGenerationError(msg)
    elif not isinstance(surface, Atoms):
        msg = f"Unrecognized input type for surface ({type(surface)})"
        raise AutocatAdsorptionGenerationError(msg)

    # Input wrangling for the different types of parameter values allowed for
    # the same function arguments.

    if adsorbates is None or not adsorbates:
        msg = "Adsorbate molecules/intermediates must be specified"
        raise AutocatAdsorptionGenerationError(msg)
    elif isinstance(adsorbates, (list, tuple)):
        adsorbates = {ads_key: ads_key for ads_key in adsorbates}
    elif not isinstance(adsorbates, dict):
        msg = f"Unrecognized input type for adsorbates ({type(adsorbates)})"
        raise AutocatAdsorptionGenerationError(msg)

    if rotations is None:
        rotations = {}
    elif isinstance(rotations, (list, tuple)):
        rotations = {ads_key: rotations for ads_key in adsorbates}
    elif not isinstance(rotations, dict):
        msg = f"Unrecognized input type for rotations ({type(rotations)})"
        raise AutocatAdsorptionGenerationError(msg)

    if adsorption_sites is None:
        adsorption_sites = {}
    elif isinstance(adsorption_sites, dict):
        # check if the input is a single site for all adsorbates vs separate
        # sites for each adsorbate
        if all([ads_key not in adsorption_sites for ads_key in adsorbates]):
            adsorption_sites = {ads_key: adsorption_sites for ads_key in adsorbates}
    elif not isinstance(adsorption_sites, dict):
        msg = f"Unrecognized input type for adsorption_sites ({type(adsorption_sites)})"
        raise AutocatAdsorptionGenerationError(msg)

    if use_all_sites is None:
        use_all_sites = {}
    elif isinstance(use_all_sites, bool):
        use_all_sites = {ads_key: use_all_sites for ads_key in adsorbates}
    elif not isinstance(use_all_sites, dict):
        msg = f"Unrecognized input type for use_all_sites ({type(use_all_sites)})"
        raise AutocatAdsorptionGenerationError(msg)

    if site_types is None:
        site_types = {}
    elif isinstance(site_types, (list, tuple)):
        site_types = {ads_key: site_types for ads_key in adsorbates}
    elif not isinstance(site_types, dict):
        msg = f"Unrecognized input type for site_types ({type(site_types)})"
        raise AutocatAdsorptionGenerationError(msg)

    if heights is None:
        heights = {}
    elif isinstance(heights, float):
        heights = {ads_key: heights for ads_key in adsorbates}
    elif not isinstance(heights, dict):
        msg = f"Unrecognized input type for heights ({type(heights)})"
        raise AutocatAdsorptionGenerationError(msg)

    if anchor_atom_indices is None:
        anchor_atom_indices = {}
    elif isinstance(anchor_atom_indices, int):
        anchor_atom_indices = {ads_key: anchor_atom_indices for ads_key in adsorbates}
    elif not isinstance(anchor_atom_indices, dict):
        msg = f"Unrecognized input type for anchor_atom_indices ({type(anchor_atom_indices)})"
        raise AutocatAdsorptionGenerationError(msg)

    ads_structures = {}
    for ads_key in adsorbates:
        ads_structures[ads_key] = {}
        # generate the molecule if not already input
        adsorbate = adsorbates.get(ads_key)
        if isinstance(adsorbate, str):
            adsorbate = generate_molecule(molecule_name=adsorbate)["structure"]
        # get all adsorption sites for the adsorbate
        ads_adsorption_sites = adsorption_sites.get(ads_key, {"origin": [(0, 0)]})
        ads_use_all_sites = use_all_sites.get(ads_key, False)
        if ads_use_all_sites:
            ads_site_types = site_types.get(ads_key, ["ontop", "bridge", "hollow"])
            ads_adsorption_sites = get_adsorption_sites(
                surface, site_types=ads_site_types
            )
        # for each adsorption site type, for each (xy) coordinate, generate the
        # adsorbated (surface + adsorbate) structure
        for site in ads_adsorption_sites:
            ads_structures[ads_key][site] = {}
            for coords in ads_adsorption_sites[site]:
                # use only xy coords and ignore z if given here (handled by ads_height)
                coords = coords[:2]
                rcoords = np.around(coords, 3)
                scoords = f"{str(rcoords[0])}_{str(rcoords[1])}"
                ads_height = heights.get(ads_key, None)
                ads_anchor_atom_index = anchor_atom_indices.get(ads_key, 0)
                ads_rotations = rotations.get(ads_key, [(0, "x")])
                adsorbed_structure = place_adsorbate(
                    surface=surface,
                    adsorbate=adsorbate,
                    adsorption_site=coords,
                    anchor_atom_index=ads_anchor_atom_index,
                    height=ads_height,
                    rotations=ads_rotations,
                )

                traj_file_path = None
                if write_to_disk:
                    dir_path = os.path.join(
                        write_location, "adsorbates", ads_key, site, scoords
                    )
                    os.makedirs(dir_path, exist_ok=dirs_exist_ok)
                    traj_file_path = os.path.join(dir_path, "input.traj")
                    adsorbed_structure.write(traj_file_path)
                    print(
                        f"Structure with {ads_key} adsorbed at {site}/{scoords}"
                        f" written to {traj_file_path}"
                    )

                ads_structures[ads_key][site].update(
                    {
                        scoords: {
                            "structure": adsorbed_structure,
                            "traj_file_path": traj_file_path,
                        }
                    }
                )

    return ads_structures


def place_adsorbate(
    surface: Atoms = None,
    adsorbate: Atoms = None,
    adsorption_site: Sequence[float] = None,
    rotations: RotationOperations = None,
    anchor_atom_index: int = 0,
    height: float = None,
) -> Atoms:
    """
    Places an adsorbate molecule/intermediate onto a given surface at the specified location.

    Parameters
    ----------

    surface (REQUIRED):
        Atoms object of the host surface.

    adsorbate (REQUIRED):
        Atoms object of the adsorbate molecule/intermediate to be placed on the
        host surface.

    adsorption_site:
        Tuple or list of the xy cartesian coordinates on the surface where the
        adsorbate must be placed.

        Defaults to [0, 0].

    rotations:
        List of rotation operations to be applied to the adsorbate
        molecule/intermediate before being placed on the host surface.

        Example:
        Rotating 90 degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [(90.0, "z"), (45.0, "y")]

        Defaults to [(0, "x")] (i.e., no rotations applied).

    anchor_atom_index:
        Integer index of the atom in the adsorbate molecule that will be used as
        anchor when placing it on the surface.
        Defaults to the atom at index 0.

    height:
        Float specifying the height above surface where adsorbate should be placed.
        If None, will estimate initial height based on covalent radii of the
        nearest neighbor atoms.

    Returns
    -------

    surface:
        Atoms object of the surface with the adsorbate molecule/intermediate
        placed on it as specified.

    """
    if adsorption_site is None:
        adsorption_site = [0, 0]

    if rotations is None:
        rotations = [(0.0, "x")]

    _surface = surface.copy()
    _adsorbate = adsorbate.copy()

    for r in rotations:
        _adsorbate.rotate(r[0], r[1])

    if height is None:
        height = get_adsorbate_height_estimate(
            _surface,
            _adsorbate,
            adsorption_site=adsorption_site,
            anchor_atom_index=anchor_atom_index,
        )

    add_adsorbate(
        _surface,
        _adsorbate,
        height,
        position=adsorption_site,
        mol_index=anchor_atom_index,
    )

    return _surface


def get_adsorbate_height_estimate(
    surface: Atoms = None,
    adsorbate: Atoms = None,
    adsorption_site: Sequence[float] = None,
    anchor_atom_index: int = 0,
    scale: float = 1.0,
) -> float:
    """
    Guess an initial height for the adsorbate to be placed on the surface, by
    summing covalent radii of the nearest neighbor atoms.

    Parameters
    ----------

    surface (REQUIRED):
        `ase.Atoms` object the surface for which adsorbate height must be estimated.

    adsorbate (REQUIRED):
        `ase.Atoms` object of adsorbate to be placed on the surface.

    adsorption_site:
        Tuple or list of the xy cartesian coordinates for where the adsorbate
        would be placed.

        Defaults to [0, 0].

    anchor_atom_index:
        Integer index of the atom in the adsorbate molecule that will be used as
        anchor when placing it on the surface.
        Defaults to the atom at index 0.

    scale:
        Float giving a scale factor to be applied to the calculated bond length.
        For example, scale = 1.1 -> bond length = 1.1 * (covalent_radius_1 + covalent_radius_2)
        Defaults to 1.0 (i.e., no scaling).

    Returns
    -------

    height_estimate:
        Float with the estimated initial height for the adsorbate molecule from
        the surface.
        Returns a default of 1.5 Angstom if the guessing process using
        nearest-neighbor covalent radii fails.
    """
    if adsorption_site is None:
        adsorption_site = (0, 0)

    species, coords = get_adsorbate_slab_nn_list(
        surface=surface, adsorption_site=adsorption_site
    )
    ads_atom_radius = covalent_radii[
        atomic_numbers[adsorbate[anchor_atom_index].symbol]
    ]
    guessed_heights = []
    for sp, coord in zip(species, coords):
        nn_radius = covalent_radii[atomic_numbers[sp]]
        r_dist = scale * (ads_atom_radius + nn_radius)
        position_array = np.array(adsorption_site)
        if (r_dist ** 2 - np.sum((position_array - coord[:2]) ** 2)) >= 0.0:
            height = np.sqrt(r_dist ** 2 - np.sum((position_array - coord[:2]) ** 2))
        else:
            height = np.nan
        guessed_heights.append(height)

    if np.isnan(np.nanmean(guessed_heights)):
        height_estimate = 1.5
    else:
        height_estimate = np.nanmean(guessed_heights)

    return height_estimate


def get_adsorbate_slab_nn_list(
    surface: Atoms = None, adsorption_site: Sequence[float] = None, height: float = 0.5
) -> Tuple[List[str], List[List[float]]]:
    """
    Get list of nearest neighbors for the adsorbate on the surface at the
    specified position.

    Parameters
    ----------

    surface (REQUIRED):
        Atoms object the surface for which nearest neighbors of the adsorbate
        should be identified.

    adsorption_site:
        Tuple or list of the xy cartesian coordinates for where the adsorbate
        would be placed.

        Defaults to [0, 0].

    height:
        Float with the height of the adsorbate molecule from the surface in
        Angstrom.

    Returns
    -------

    species_list, coordinates_list:
        Two lists of length equal to the number of nearest neighbors identified:
        first with the list of species names, and second with the list of
        coordinates.

        Example:
        (["Fe", "Sr"], [[0.4, 0.6], [0.2, 0.9]])
    """
    if adsorption_site is None:
        adsorption_site = (0, 0)

    surf = surface.copy()
    conv = AseAtomsAdaptor()
    add_adsorbate(surf, "X", height=height, position=adsorption_site)
    init_guess = conv.get_structure(surf)
    nn = get_neighbors_of_site_with_index(init_guess, -1)
    species_list = [_nn.species.hill_formula for _nn in nn]
    coord_list = [_nn.coords for _nn in nn]
    return species_list, coord_list


def get_adsorption_sites(
    surface: Atoms = None, site_types: Sequence[str] = None, **kwargs
) -> Dict[str, Sequence[float]]:
    """
    For the given surface, returns all symmetrically unique sites of the
    specified type.

    Uses `pymatgen.analysis.adsorption` module to identify the sites.

    Parameters
    ----------

    surface (REQUIRED):
        Atoms object of the host surface for which the symmetrically unique
        surface sites should be identified.

    site_types:
        List of types of adsorption sites to be searched for. Options are
        "ontop", "bridge", and "hollow".

        Defaults to ["ontop", "bridge", "hollow"].

    **kwargs:
        Other miscellaneous keyword arguments. Will be passed on to
        `AdsorbateSiteFinder.find_adsorption_sites` in `pymatgen`.

    Returns
    -------

    sites:
        Dictionary containing the reference structures with the identified
        sites for each desired site type
    """
    if site_types is None:
        site_types = ["ontop", "bridge", "hollow"]

    converter = AseAtomsAdaptor()
    pmg_structure = converter.get_structure(surface)
    finder = AdsorbateSiteFinder(pmg_structure)
    sites = finder.find_adsorption_sites(positions=site_types, **kwargs)
    # Remove the extraneous "all" returned by default from pymatgen
    sites.pop("all", None)

    return sites
