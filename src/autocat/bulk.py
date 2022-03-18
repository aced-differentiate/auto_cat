import os
from typing import List
from typing import Dict
from typing import Any

from ase.build import bulk
from ase.data import atomic_numbers
from ase.data import reference_states
from ase.data import ground_state_magnetic_moments
from autocat.data.lattice_parameters import BULK_PBE_FD
from autocat.data.lattice_parameters import BULK_PBE_PW
from autocat.data.lattice_parameters import BULK_BEEFVDW_FD
from autocat.data.lattice_parameters import BULK_BEEFVDW_PW


def generate_bulk_structures(
    species_list: List[str],
    crystal_structures: Dict[str, str] = None,
    default_lat_param_lib: str = None,
    a_dict: Dict[str, float] = None,
    c_dict: Dict[str, float] = None,
    set_magnetic_moments: List[str] = None,
    magnetic_moments: Dict[str, float] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Generates bulk crystal structures and writes them to separate
    directories, if specified.

    Parameters
    ----------

    species_list (REQUIRED):
        List of chemical symbols of the bulk structures to be constructed.

    cystal_structures:
        Dictionary with crystal structure to be used for each species.
        These will be passed on as input to `ase.build.bulk`. So, must be one
        of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic,
        diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite.
        If not specified, the default reference crystal structure for each
        species from `ase.data` will be used.

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
        reference library specified, it will be pulled from `ase.data`.

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

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.

        In the specified write_location, the following directory structure
        will be created:
        [species_1]_bulk_[crystal_structure_1]/input.traj
        [species_1]_bulk_[crystal_structure_2]/input.traj
        ...
        [species_2]_bulk_[crystal_structure_2]/input.traj
        ...
        Defaults to the current working directory.

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    Dictionary with bulk structures (as `ase.Atoms` objects) and
    write-location (if any) for each input species.

    Example:

    {
        "Pt": {"structure": Pt_ase_obj, "traj_file_path": "/path/to/Pt/traj/file"},
        "Cr": ...
    }

    """

    lpl = {
        "pbe_fd": BULK_PBE_FD,
        "beefvdw_fd": BULK_BEEFVDW_FD,
        "pbe_pw": BULK_PBE_PW,
        "beefvdw_pw": BULK_BEEFVDW_PW,
    }

    if crystal_structures is None:
        crystal_structures = {}
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

    bulk_structures = {}
    for species in species_list:
        cs = cs_library.get(species)
        a = a_library.get(species)
        c = c_library.get(species)

        bs = bulk(species, crystalstructure=cs, a=a, c=c)

        if species in set_magnetic_moments:
            bs.set_initial_magnetic_moments([mm_library[species]] * len(bs))

        traj_file_path = None
        if write_to_disk:
            dir_path = os.path.join(write_location, f"{species}_bulk_{cs}")
            os.makedirs(dir_path, exist_ok=dirs_exist_ok)
            traj_file_path = os.path.join(dir_path, "input.traj")
            bs.write(traj_file_path)
            print(f"{species}_bulk_{cs} structure written to {traj_file_path}")

        bulk_structures[species] = {
            "structure": bs,
            "traj_file_path": traj_file_path,
        }

    return bulk_structures
