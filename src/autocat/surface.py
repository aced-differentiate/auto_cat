import os
from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import Union

from ase.io import read, write
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from ase.build import hcp0001
from ase.data import reference_states, atomic_numbers, ground_state_magnetic_moments
from ase.constraints import FixAtoms


def generate_surface_structures(
    species_list: List[str],
    crystal_structures: Dict[str, str] = None,
    ft_dict: Dict[str, str] = None,
    supcell: Union[Tuple[int], List[int]] = (3, 3, 4),
    a_dict: Optional[Dict[str, float]] = None,
    c_dict: Optional[Dict[str, float]] = None,
    set_magnetic_moments: List[str] = None,
    magnetic_moments: Optional[Dict[str, float]] = None,
    vac: float = 10.0,
    fix: int = 0,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    Generates mono-element slabs and writes them to separate directories,
    if specified.

    Parameters
    ----------

    species_list:
        List of chemical symbols of the slabs to be built

    crystal_structures:
        Dictionary with crystal structure to be used for each species.
        Options are fcc, bcc, or hcp. If not specified, will use the
        default reference crystal for each species from `ase.data`.
    
    ft_dict:
        Dictionary with the surface facets to be considered for each
        species. 

        If not specified for a given species, the following
        defaults will be used based on the crystal structure:
        fcc/bcc: 100,111,110
        hcp: 0001
        
    supcell: 
        Tuple or List specifying the size of the supercell to be
        generated in the format (nx,ny,nz).

    a_dict:
        Dictionary with lattice parameters <a> to be used for each species.
        If not specified, defaults from the `ase.data` module are used.

    c_dict:
        Dictionary with lattice parameters <c> to be used for each species.
        If not specified, defaults from the `ase.data` module are used.

    set_magnetic_moments:
        List of species for which magnetic moments need to be set.
        If not specified, magnetic moments will be set only for Fe, Co, Ni
        (the ferromagnetic elements).

    magnetic_moments:
        Dictionary with the magnetic moments to be set for the chemical
        species listed previously.
        If not specified, default ground state magnetic moments from
        `ase.data` are used. 

    vac:
        Float specifying the amount of vacuum to be added on each
        side of the slab.

    fix:
        Integer giving the number of layers of the slab to be fixed
        starting from the bottom up. (e.g. a value of 2 will fix the
        bottom 2 layers)

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

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------
 
    Dictionary with surface structures (as `ase.Atoms` objects) and
    write-location (if any) for each input species.

    """

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
    if ft_dict is None:
        ft_dict = {}

    # load crystal structure defaults from `ase.data`, override with user input
    cs_library = {
        species: reference_states[atomic_numbers[species]].get("symmetry")
        for species in species_list
    }
    cs_library.update(crystal_structures)

    # load magnetic moment defaults from `ase.data`, override with user input
    mm_library = {
        species: ground_state_magnetic_moments[atomic_numbers[species]]
        for species in species_list
    }
    mm_library.update(magnetic_moments)

    funcs = {
        "fcc100": fcc100,
        "fcc110": fcc110,
        "fcc111": fcc111,
        "bcc100": bcc100,
        "bcc110": bcc110,
        "bcc111": bcc111,
        "hcp0001": hcp0001,
    }

    # set default facets for each crystal structure, override with user input
    ft_defaults = {
        "fcc": ["100", "111", "110"],
        "bcc": ["100", "111", "110"],
        "hcp": ["0001"],
    }

    ft_library = {species: ft_defaults[cs_library[species]] for species in species_list}
    ft_library.update(ft_dict)

    surface_structures = {}
    for species in species_list:
        cs = cs_library.get(species)
        a = a_dict.get(species)
        c = c_dict.get(species)

        surf = {}
        for facet in ft_library[species]:
            if c is not None:
                struct = funcs[cs + facet](species, size=supcell, vacuum=vac, a=a, c=c)
            else:
                struct = funcs[cs + facet](species, size=supcell, vacuum=vac, a=a)

            if fix > 0:
                f = FixAtoms(mask=[atom.tag > (supcell[-1] - fix) for atom in struct])
                struct.set_constraint([f])

            if species in set_magnetic_moments:
                struct.set_initial_magnetic_moments([mm_library[species]] * len(struct))

            traj_file_path = None
            if write_to_disk:
                dir_path = os.path.join(write_location, f"{species}/{cs}{facet}")
                os.makedirs(dir_path, exist_ok=dirs_exist_ok)
                traj_file_path = os.path.join(dir_path, "input.traj")
                struct.write(traj_file_path)
                print(f"{species}_{cs}{facet} structure written to {traj_file_path}")

            surf[cs + facet] = {"structure": struct, "traj_file_path": traj_file_path}

        surface_structures[species] = surf
    return surface_structures
