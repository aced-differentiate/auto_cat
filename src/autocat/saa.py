import os
import numpy as np
from typing import List
from typing import Dict
from typing import Any
from typing import Sequence

from ase import Atoms
from ase.data import atomic_numbers
from ase.data import ground_state_magnetic_moments
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from autocat.surface import generate_surface_structures


class AutocatSaaGenerationError(Exception):
    pass


def _find_dopant_index(structure, dopant_element):
    """Helper function for finding the index of the (single) dopant atom."""
    # TODO(@lancekavalsky): implement multi-dopant-atom indices
    #    # Find index of species with lowest count
    #    unique, counts = np.unique(syms, return_counts=True)
    #    ind = np.where(syms == unique[np.argmin(counts)])[0][0]
    symbols = np.array(structure.symbols)
    dopant_index = np.where(symbols == dopant_element)
    if np.size(dopant_index) < 1:
        msg = f"Dopant element {dopant_element} not found in structure"
        raise AutocatSaaGenerationError(msg)
    elif np.size(dopant_index) > 1:
        msg = f"More than one atom of {dopant_element} found in structure"
        raise NotImplementedError(msg)
    return dopant_index[0][0]


def _find_all_surface_atom_indices(structure, tol: float = 0.5) -> List[int]:
    """Helper function to find all surface atom indices
    within a tolerance distance of the highest atom"""
    all_heights = structure.positions[:, 2]
    highest_atom_idx = np.argmax(all_heights)
    height_of_highest_atom = structure[highest_atom_idx].z
    surface_atom_indices = []
    for idx, atom in enumerate(structure):
        if height_of_highest_atom - atom.z < tol:
            surface_atom_indices.append(idx)
    return surface_atom_indices


def generate_saa_structures(
    host_species: List[str],
    dopant_species: List[str],
    crystal_structures: Dict[str, str] = None,
    facets: Dict[str, str] = None,
    supercell_dim: Sequence[int] = (3, 3, 4),
    default_lat_param_lib: str = None,
    a_dict: Dict[str, float] = None,
    c_dict: Dict[str, float] = None,
    set_host_magnetic_moments: List[str] = None,
    host_magnetic_moments: Dict[str, float] = None,
    set_dopant_magnetic_moments: List[str] = None,
    dopant_magnetic_moments: Dict[str, float] = None,
    vacuum: float = 10.0,
    n_fixed_layers: int = 0,
    place_dopant_at_center: bool = True,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Builds single-atom alloys for all combinations of host species and dopant
    species given. Will write the structures to separate directories if
    specified.

    Parameters
    ----------

    host_species (REQUIRED):
        List of chemical species of desired host (substrate) species.

    dopant_species (REQUIRED):
        List of chemical symbols of desired single-atom dopant species.

    crystal_structures:
        Dictionary with crystal structure to be used for each species.
        These will be passed on as input to `ase.build.bulk`. So, must be one
        of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic,
        diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite.
        If not specified, the default reference crystal structure for each
        species from `ase.data` will be used.

    facets:
        Dictionary with the surface facets to be considered for each
        species.
        If not specified for a given species, the following defaults will be
        used based on the crystal structure:
        fcc/bcc: 100, 111, 110
        hcp: 0001

    supercell_dim:
        Tuple or List specifying the size of the supercell to be
        generated in the format (nx, ny, nz).
        Defaults to (3, 3, 4).

    default_lat_param_lib:
        String indicating which library the lattice constants should be pulled
        from if not specified in either a_dict or c_dict.

        Options:
        pbe_fd: parameters calculated using xc=PBE and finite-difference
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and finite-difference
        pbe_pw: parameters calculated using xc=PBE and a plane-wave basis set
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and a plane-wave basis set

        N.B. if there is a species present in `host_species` that is NOT in the
        reference library specified, it will be pulled from `ase.data`.

    a_dict:
        Dictionary with lattice parameters <a> to be used for each species.
        If not specified, defaults from `default_lat_param_lib` are used.

    c_dict:
        Dictionary with lattice parameters <c> to be used for each species.
        If not specified, defaults from `default_lat_param_lib` are used.

    set_host_magnetic_moments:
        List of host species for which magnetic moments need to be set.
        If not specified, magnetic moments will be set only for Fe, Co, Ni
        (the ferromagnetic elements).

    host_magnetic_moments:
        Dictionary with the magnetic moments to be set for the host chemical
        species listed previously.
        If not specified, default ground state magnetic moments from
        `ase.data` are used.

    set_dopant_magnetic_moments:
        List of single-atom species for which magnetic moments need to be set.
        If not specified, magnetic moments will guessed for all dopant species from
        `ase.data`.

    dopant_magnetic_moments:
        Dictionary with the magnetic moments to be set for the single-atom
        dopant species listed previously.
        If not specified, default ground state magnetic moments from
        `ase.data` are used.

    vacuum:
        Float specifying the amount of vacuum (in Angstrom) to be added to
        the slab (the slab is placed at the center of the supercell).
        Defaults to 10.0 Angstrom.

    n_fixed_layers:
        Integer giving the number of layers of the slab to be fixed
        starting from the bottom up (e.g., a value of 2 will fix the
        bottom 2 layers).
        Defaults to 0 (i.e., no layers in the slab fixed).

    place_dopant_at_center:
        Boolean specifying whether the single-atom should be placed
        at the center of the unit cell. If False, the single-atom will
        be placed at the origin.
        Defaults to True.

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.
        In the specified write_location, the following directory structure
        will be created:
        [host]/[dopant]/[facet]/substrate/input.traj

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    Dictionary with the single-atom alloy structures as `ase.Atoms` objects and
    write-location, if any, for each {crystal structure and facet} specified for
    each input host and dopant species combination.

    Example:
    {
        "Fe": {
            "Cu": {
                "bcc100": {
                    "structure": FeN-1_Cu1_saa_obj,
                    "traj_file_path": "/path/to/Cu/on/bcc/Fe/100/surface/traj/file"
                },
                "bcc110": ...,
            },
            "Ru": {
                ...
            },
        },
        "Rh": {
            ...
        }
    }

    """

    hosts = generate_surface_structures(
        host_species,
        crystal_structures=crystal_structures,
        facets=facets,
        supercell_dim=supercell_dim,
        default_lat_param_lib=default_lat_param_lib,
        a_dict=a_dict,
        c_dict=c_dict,
        set_magnetic_moments=set_host_magnetic_moments,
        magnetic_moments=host_magnetic_moments,
        vacuum=vacuum,
        n_fixed_layers=n_fixed_layers,
    )

    if set_dopant_magnetic_moments is None:
        set_dopant_magnetic_moments = dopant_species
    if dopant_magnetic_moments is None:
        dopant_magnetic_moments = {}

    dop_mm_library = {
        dop: ground_state_magnetic_moments[atomic_numbers[dop]]
        for dop in dopant_species
    }
    dop_mm_library.update(dopant_magnetic_moments)

    saa_structures = {}
    # iterate over hosts
    for host in hosts:
        saa_structures[host] = {}
        # iterate over single-atoms
        for dopant in dopant_species:
            # ensure host != single-atom
            if dopant == host:
                continue
            saa_structures[host][dopant] = {}
            # iterate over surface facets
            for facet in hosts[host]:
                host_structure = hosts[host][facet].get("structure")
                doped_structure = substitute_single_atom_on_surface(
                    host_structure,
                    dopant,
                    place_dopant_at_center=place_dopant_at_center,
                    dopant_magnetic_moment=dop_mm_library.get(dopant),
                )

                traj_file_path = None
                if write_to_disk:
                    dir_path = os.path.join(
                        write_location, host, dopant, facet, "substrate"
                    )
                    os.makedirs(dir_path, exist_ok=dirs_exist_ok)
                    traj_file_path = os.path.join(dir_path, "input.traj")
                    doped_structure.write(traj_file_path)
                    print(
                        f"{dopant}/{host}({facet}) structure written to {traj_file_path}"
                    )

                saa_structures[host][dopant][facet] = {
                    "structure": doped_structure,
                    "traj_file_path": traj_file_path,
                }
    return saa_structures


def substitute_single_atom_on_surface(
    host_structure: Atoms,
    dopant_element: str,
    place_dopant_at_center: bool = True,
    dopant_magnetic_moment: float = 0.0,
) -> Atoms:
    """
    For a given host (**elemental surface**) structure and a dopant element,
    returns a slab with one host atom on the surface substituted with the
    specified dopant element with a specified magnetic moment.
    Note that for the current implementation (single-atom alloys), there will
    exist only one symmetrically unique site to substitute on the surface of the
    elemental slab.

    Parameters
    ----------

    host_structure (REQUIRED):
        ase.Atoms object of the host slab to be doped.

    dopant_element (REQUIRED):
        String of the elemental species to be substitutionally doped into the
        host structure.

    place_dopant_at_center:
        Boolean specifying whether the single-atom dopant should be placed at
        the center of the unit cell. If False, the dopant atom will be placed at
        the origin.
        Defaults to True.

    dopant_magnetic_moment:
        Float with the initial magnetic moment on the doped single-atom.
        Defaults to no spin polarization (i.e., magnetic moment of 0).

    Returns
    -------

    The elemental slab with a single-atom dopant on the surface as an
    `ase.Atoms` object.

    Raises
    ------

    NotImplementedError
        If multiple symmetrically equivalent sites are found on the surface to dope.
        Note that is intended more as a "guardrail" on current functionality to
        match the maturity/implementation of other modules in `autocat` than an
        actual error. The error should no longer be present when the
        substitution functionality is folded into a more general form.

    """

    all_surface_indices = _find_all_surface_atom_indices(host_structure)

    ase_all_doped_structures = []
    for idx in all_surface_indices:
        dop_struct = host_structure.copy()
        dop_struct[idx].symbol = dopant_element
        dop_struct[idx].magmom = dopant_magnetic_moment
        ase_all_doped_structures.append(dop_struct)

    # convert ase substrate to pymatgen structure
    converter = AseAtomsAdaptor()
    pmg_doped_structures = [
        converter.get_structure(struct) for struct in ase_all_doped_structures
    ]

    # check that only one unique surface doped structure
    matcher = StructureMatcher()
    pmg_symm_equiv_doped_structure = [
        s[0] for s in matcher.group_structures(pmg_doped_structures)
    ]
    if len(pmg_symm_equiv_doped_structure) > 1:
        msg = "Multiple symmetrically unique sites to dope found."
        raise NotImplementedError(msg)

    # assumes only a single unique doped structure
    ase_substituted_structure = ase_all_doped_structures[0]

    # center the single-atom dopant
    if place_dopant_at_center:
        cent_x = (
            ase_substituted_structure.cell[0][0] / 2
            + ase_substituted_structure.cell[1][0] / 2
        )
        cent_y = (
            ase_substituted_structure.cell[0][1] / 2
            + ase_substituted_structure.cell[1][1] / 2
        )
        cent = (cent_x, cent_y, 0)
        ase_substituted_structure.translate(cent)
        ase_substituted_structure.wrap()

    return ase_substituted_structure
