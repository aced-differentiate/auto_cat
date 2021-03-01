import os
import numpy as np
from typing import List
from typing import Dict
from collections.abc import Sequence

from ase import Atoms
from ase.data import atomic_numbers
from ase.data import ground_state_magnetic_moments
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from autocat.surface import generate_surface_structures


class AutocatStructureGenerationError(Exception):
    pass


# TODO(@hegdevinayi): Typing for the returned data
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
):
    """
    Builds single-atom alloys for all combinations of host species and dopant
    species given. Will write the structures to separate directories if
    specified.

    Parameters
    ----------

    host_species (REQUIRED):
        List of chemical species of desired host species.

    dopant_species (REQUIRED):
        List of chemical symbols of desired single-atom dopant species.

    crystal_structures:
        Dictionary with crystal structure to be used for each species.
        Options are fcc, bcc, or hcp.
        If not specified, will use the default reference crystal for each
        species from `ase.data`.

    facets:
        Dictionary with the surface facets to be considered for each
        species.
        If not specified for a given species, the following
        defaults will be used based on the crystal structure:
        fcc/bcc: 100, 111, 110
        hcp: 0001

    supercell_dim:
        Tuple or List specifying the size of the supercell to be
        generated in the format (nx, ny, nz).
        Defaults to (3, 3, 4).

    default_lat_param_lib:
        String indicating which library the lattice constants should be pulled
        from if not specified in either a_dict or c_dict.

        Options are:
        pbe_fd: parameters calculated using xc=pbe and finite-difference
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and finite-difference
        pbe_pw: parameters calculated using xc=pbe and a plane-wave basis set
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and a plane-wave basis set

        N.B. if there is a species present in species_list that is NOT in the
        reference library specified, it will be pulled from `ase.data`

    a_dict:
        Dictionary with lattice parameters <a> to be used for each species.
        If not specified, defaults from default_lat_param_lib are used.

    c_dict:
        Dictionary with lattice parameters <c> to be used for each species.
        If not specified, defaults from default_lat_param_lib are used.

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
        If not specified, magnetic moments will guessed for all dopant_species from
        `ase.data`.

    dopant_magnetic_moments:
        Dictionary with the magnetic moments to be set for the single-atom
        dopant species listed previously.
        If not specified, default ground state magnetic moments from
        `ase.data` are used.

    vacuum:
        Float specifying the amount of vacuum (in Angstrom) to be added to
        the slab (the slab is placed at the center of the supercell).

    n_fixed_layers:
        Integer giving the number of layers of the slab to be fix
        starting from the bottom up (e.g. a value of 2 will fix the
        bottom 2 layers).

    place_dopant_at_center:
        Boolean specifying whether the single-atom should be placed
        at the center of the unit cell. If False, the single-atom will
        be placed at the origin.

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

    Dictionary containing the generated single-atom alloy structures.
    Organized by host -> dopant -> facet.
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

    saa_dict = {}
    # iterate over hosts
    for host in hosts:
        saa_dict[host] = {}
        # iterate over single-atoms
        for dopant in dopant_species:
            # ensure host != single-atom
            if dopant == host:
                continue
            saa_dict[host][dopant] = {}
            # iterate over surface facets
            for facet in hosts[host]:
                host_structure = hosts[host][facet].get("structure")
                doped_structures = substitute_dopant_on_surface(
                    host_structure=host_structure,
                    dopant=dopant,
                    place_dopant_at_center=place_dopant_at_center,
                    dopant_magnetic_moment=dop_mm_library.get(dopant),
                    write_to_disk=False,
                )
                # check that there is only one substituted structure (SAA)
                assert len(doped_structures) == 1
                doped_structure = list(doped_structures.values())[0]["structure"]

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

                saa_dict[host][dopant][facet] = {
                    "structure": doped_structure,
                    "traj_file_path": traj_file_path,
                }
    return saa_dict


# TODO: generalize this function/reimplement for custom substitutions
def substitute_dopant_on_surface(
    host_structure: Atoms,
    dopant_element: str,
    place_dopant_at_center: bool = True,
    dopant_magnetic_moment: float = 0.0,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
    **kwargs: str,
):
    """
    Generates doped structures given host (**surface**) structure and a
    dopant element. Uses pymatgen's `AdsorbateSiteFinder` module to find all
    symmetrically unique sites to substitute on.

    If specified, will write to separate directories for each generated doped
    system organized by target indices.

    Parameters
    ----------

    host_structure:
        ase.Atoms object of the host slab to be doped.

    dopant_element:
        String of the elemental species to be substitutionally doped into the
        host structure.

    place_dopant_at_center:
        Boolean specifying that the single-atom dopant should be placed
        at the center of the unit cell if True.

    dopant_magnetic_moment:
        Float of initial magnetic moment attributed to the doped single-atom.
        Will default to no spin polarization (i.e., magnetic moment of 0).

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.

        In the specified write_location, the following directory structure
        will be created:
        [host]_[dopant]_[atom index substituted]/input.traj

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    Dictionary with doped structures (as `ase.Atoms` objects) and write
    location (if any) for each generated doped structure.

    """
    name = "".join(np.unique(host_structure.symbols))
    tags = host_structure.get_tags()
    constraints = host_structure.constraints
    host_magmom = host_structure.get_initial_magnetic_moments()

    # convert ase substrate to pymatgen structure
    converter = AseAtomsAdaptor()  # converter between pymatgen and ase
    pmg_structure = converter.get_structure(host_structure)

    # find all symmetrically unique site to substitute on
    finder = AdsorbateSiteFinder(pmg_structure)

    # collect all substitution structures
    pmg_substituted_structures = finder.generate_substitution_structures(dopant_element)
    ase_substituted_structures = [
        converter.get_atoms(s) for s in pmg_substituted_structures
    ]

    substituted_structures = {}

    for struct in ase_substituted_structures:
        struct.set_tags(tags)
        struct.pbc = (1, 1, 0)  # ensure pbc in xy only
        struct.constraints = constraints  # propagate constraints
        struct.set_initial_magnetic_moments(host_magmom)  # propagate host magnetization
        dopant_idx = _find_dopant_index(struct, dopant_element)
        struct[dopant_idx].magmom = dopant_magnetic_moment  # set initial magmom
        if place_dopant_at_center:  # centers the sa
            cent_x = struct.cell[0][0] / 2 + struct.cell[1][0] / 2
            cent_y = struct.cell[0][1] / 2 + struct.cell[1][1] / 2
            cent = (cent_x, cent_y, 0)
            struct.translate(cent)
            struct.wrap()

        traj_file_path = None
        if write_to_disk:
            dir_path = os.path.join(
                write_location, name + "_" + dopant_element + "_" + str(dopant_idx)
            )
            os.makedirs(dir_path, exist_ok=dirs_exist_ok)
            traj_file_path = os.path.join(dir_path, "input.traj")
            struct.write(traj_file_path)
            print(
                f"{name}_{dopant_element}_{str(dopant_idx)} structure written to {traj_file_path}"
            )

        substituted_structures[str(dopant_idx)] = {
            "structure": struct,
            "traj_file_path": traj_file_path,
        }

    return substituted_structures


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
        raise AutocatStructureGenerationError(msg)
    elif np.size(dopant_index) > 1:
        msg = f"More than one atom of {dopant_element} found in structure"
        raise NotImplementedError(msg)
    return dopant_index[0][0]
