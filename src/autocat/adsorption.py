import os
import numpy as np
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
