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
    Given list of species, bravais lattice, and facets creates directories containing traj files for the surfaces

    Parameters:
        species_list(list of str): list of surf species to be generated
        cs(dict): dict of manually specified bravais lattices for specific species
        ft_dict(dict): facets to be considered for each species
        supcell(tuple of int): supercell size to be generated
        a(dict): manually specified lattice parameters for species. if None then uses ASE default
        c(dict): manually specified lattice parameters for species. if None then uses ASE default

    Returns:
        None
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

            surf[cs + facet] = {"structure": struct, "traj_file_path": traj_file_path}

        surface_structures[species] = surf
    return surface_structures


def gen_surf(
    species,
    cs=None,
    ft=["100", "110", "111"],
    supcell=(3, 3, 4),
    a=None,
    c=None,
    fix=0,
    write_traj=False,
):
    """
    Given species, bravais lattice, and facets, generates dict of ase objects for surfaces

    Parameters
    species (str): surf species
    cs (str): bravais lattice
    ft (list of str): facets to be considered
    supcell (tuple): supercell size
    a (float): lattice parameter. if None uses ASE default
    c (float): lattice parameter. if None uses ASE default
    fix (int): number of layers from bottom to fix (e.g. value of 2 fixes bottom 2 layers)

    Returns
    surf (dict): dictionary of generated surf facets
    """

    if cs is None:  # uses ASE data to get Bravais Lattice
        cs = reference_states[atomic_numbers[species]]["symmetry"]

    surf = {}
    funcs = {
        "fcc100": fcc100,
        "fcc110": fcc110,
        "fcc111": fcc111,
        "bcc100": bcc100,
        "bcc110": bcc110,
        "bcc111": bcc111,
        "hcp0001": hcp0001,
    }
    j = 0
    while j < len(ft):
        if c is not None:
            surf[cs + ft[j]] = funcs[cs + ft[j]](
                species, size=supcell, vacuum=10.0, a=a, c=c
            )
        else:
            surf[cs + ft[j]] = funcs[cs + ft[j]](
                species, size=supcell, vacuum=10.0, a=a
            )
        j += 1
    if fix > 0:
        for sys in surf:
            f = FixAtoms(mask=[atom.tag > (supcell[-1] - fix) for atom in surf[sys]])
            surf[sys].set_constraint([f])

    if species in ["Fe", "Co", "Ni"]:  # check if ferromagnetic material
        for sys in surf:
            mag = ground_state_magnetic_moments[atomic_numbers[species]]
            surf[sys].set_initial_magnetic_moments([mag] * len(surf[sys]))

    if write_traj:
        for sys in surf:
            surf[sys].write(species + "_" + sys + ".i.traj")
    return surf
