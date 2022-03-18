import os
from typing import List
from typing import Dict
from typing import Any

import ase.build
from ase.data import reference_states
from ase.data import atomic_numbers
from ase.data import ground_state_magnetic_moments
from ase.constraints import FixAtoms
from autocat.data.lattice_parameters import BULK_PBE_FD
from autocat.data.lattice_parameters import BULK_PBE_PW
from autocat.data.lattice_parameters import BULK_BEEFVDW_FD
from autocat.data.lattice_parameters import BULK_BEEFVDW_PW


def generate_surface_structures(
    species_list: List[str],
    crystal_structures: Dict[str, str] = None,
    facets: Dict[str, str] = None,
    supercell_dim: List[int] = None,
    default_lat_param_lib: str = None,
    a_dict: Dict[str, float] = None,
    c_dict: Dict[str, float] = None,
    set_magnetic_moments: List[str] = None,
    magnetic_moments: Dict[str, float] = None,
    vacuum: float = 10.0,
    n_fixed_layers: int = 0,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generates mono-element slabs and writes them to separate directories,
    if specified.

    Parameters
    ----------

    species_list (REQUIRED):
        List of chemical symbols of the slabs to be built.

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

        If not specified for a given species, the following
        defaults will be used based on the crystal structure:
        fcc/bcc: 100, 111, 110
        hcp: 0001

    supercell_dim:
        List specifying the size of the supercell to be generated in the
        format (nx, ny, nz).

    default_lat_param_lib:
        String indicating which library the lattice constants should be pulled
        from if not specified in either a_dict or c_dict.
        Defaults to lattice constants defined in `ase.data`.

        Options:
        pbe_fd: parameters calculated using xc=PBE and finite-difference
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and finite-difference
        pbe_pw: parameters calculated using xc=PBE and a plane-wave basis set
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and a plane-wave basis set

        N.B. if there is a species present in species_list that is NOT in the
        reference library specified, it will be pulled from `ase.data`

    a_dict:
        Dictionary with lattice parameters <a> to be used for each species.
        If not specified, defaults from `default_lat_param_lib` are used.

    c_dict:
        Dictionary with lattice parameters <c> to be used for each species.
        If not specified, defaults from `default_lat_param_lib` are used.

    set_magnetic_moments:
        List of species for which magnetic moments need to be set.
        If not specified, magnetic moments will be set only for Fe, Co, Ni
        (the ferromagnetic elements).

    magnetic_moments:
        Dictionary with the magnetic moments to be set for the chemical
        species listed previously.
        If not specified, default ground state magnetic moments from
        `ase.data` are used.

    vacuum:
        Float specifying the amount of vacuum (in Angstrom) to be added to
        the slab (the slab is placed at the center of the supercell).

    n_fixed_layers:
        Integer giving the number of layers of the slab to be fixed
        starting from the bottom up (e.g. a value of 2 will fix the
        bottom 2 layers).

    write_to_disk:
        Boolean specifying whether the surface structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.

        In the specified write_location, the following directory structure
        will be created:
        [species]/[crystal_structure + facet]/substrate/input.traj
        Defaults to the current working directory.

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    Dictionary with surface structures (as `ase.Atoms` objects) and
    write-location (if any) for each {crystal structure and facet} specified for
    each input species.

    Example:

    {
        "Pt": {
            "fcc111": {
                "structure": Pt_surface_obj,
                "traj_file_path": "/path/to/Pt/fcc111/surface/traj/file"
                },
            "fcc100": ...
            ,
        "Cu": ...
        }
    }

    """

    lpl = {
        "pbe_fd": BULK_PBE_FD,
        "beefvdw_fd": BULK_BEEFVDW_FD,
        "pbe_pw": BULK_PBE_PW,
        "beefvdw_pw": BULK_BEEFVDW_PW,
    }
    ase_build_funcs = {
        "fcc100": ase.build.fcc100,
        "fcc110": ase.build.fcc110,
        "fcc111": ase.build.fcc111,
        "bcc100": ase.build.bcc100,
        "bcc110": ase.build.bcc110,
        "bcc111": ase.build.bcc111,
        "hcp0001": ase.build.hcp0001,
    }

    if crystal_structures is None:
        crystal_structures = {}
    if facets is None:
        facets = {}
    if supercell_dim is None:
        supercell_dim = [3, 3, 4]
    if a_dict is None:
        a_dict = {}
    if c_dict is None:
        c_dict = {}
    if set_magnetic_moments is None:
        set_magnetic_moments = ["Fe", "Co", "Ni"]
    if magnetic_moments is None:
        magnetic_moments = {}

    # load crystal structure defaults from `ase.data`, override with user input
    cs_library = {
        species: reference_states[atomic_numbers[species]].get("symmetry")
        for species in species_list
    }
    cs_library.update(crystal_structures)

    # load lattice params <a>, <c> from reference library, override with user input
    a_library = {}
    c_library = {}
    if default_lat_param_lib is not None:
        a_library.update(
            {
                species: lpl[default_lat_param_lib].get(species, {}).get("a")
                for species in species_list
            }
        )
        c_library.update(
            {
                species: lpl[default_lat_param_lib].get(species, {}).get("c")
                for species in species_list
            }
        )
    a_library.update(a_dict)
    c_library.update(c_dict)

    # load magnetic moment defaults from `ase.data`, override with user input
    mm_library = {
        species: ground_state_magnetic_moments[atomic_numbers[species]]
        for species in species_list
    }
    mm_library.update(magnetic_moments)

    # set default facets for each crystal structure, override with user input
    ft_defaults = {
        "fcc": ["100", "111", "110"],
        "bcc": ["100", "111", "110"],
        "hcp": ["0001"],
    }

    ft_library = {species: ft_defaults[cs_library[species]] for species in species_list}
    ft_library.update(facets)

    surface_structures = {}
    for species in species_list:
        cs = cs_library.get(species)
        a = a_library.get(species)
        c = c_library.get(species)

        surf = {}
        for facet in ft_library[species]:
            if c is not None:
                struct = ase_build_funcs[f"{cs}{facet}"](
                    species, size=supercell_dim, vacuum=vacuum, a=a, c=c
                )
            else:
                struct = ase_build_funcs[f"{cs}{facet}"](
                    species, size=supercell_dim, vacuum=vacuum, a=a
                )

            if n_fixed_layers > 0:
                f = FixAtoms(
                    mask=[
                        atom.tag > (supercell_dim[-1] - n_fixed_layers)
                        for atom in struct
                    ]
                )
                struct.set_constraint([f])

            if species in set_magnetic_moments:
                struct.set_initial_magnetic_moments([mm_library[species]] * len(struct))

            traj_file_path = None
            if write_to_disk:
                dir_path = os.path.join(
                    write_location, f"{species}", f"{cs}{facet}", "substrate"
                )
                os.makedirs(dir_path, exist_ok=dirs_exist_ok)
                traj_file_path = os.path.join(dir_path, "input.traj")
                struct.write(traj_file_path)
                print(f"{species}_{cs}{facet} structure written to {traj_file_path}")

            surf[f"{cs}{facet}"] = {
                "structure": struct,
                "traj_file_path": traj_file_path,
            }

        surface_structures[species] = surf
    return surface_structures
