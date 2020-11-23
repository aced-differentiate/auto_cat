import os
import numpy as np
from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import Union

from ase.io import read, write
from ase import Atom, Atoms
from ase.visualize import view
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from ase.data import reference_states, atomic_numbers
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

from autocat.surface import generate_surface_structures


def generate_mpea_random(
    species_list: List[str],
    composition: Dict[str, float] = None,
    lattice_parameters: Dict[str, float] = None,
    crystal_structure: str = "fcc",
    facets: List[str] = None,
    supcell: Union[Tuple[int], List[int]] = (3, 3, 4),
    num_of_samples: int = 15,
    vac: float = 10.0,
    fix: int = 0,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """

    For the given species list and composition will generate a specified number of samples from this space.
    If specified will write these samples in separate directories

    Parameters
    ----------

    species_list:
        List of species that will populate the skeleton structure

    composition:
        Dictionary of desired composition, defaults to be 1/sum(composition) for each species
        not mentioned. Used for calculating probabilities of each species occupying a given site.

        e.g. species_list = ["Pt","Fe","Cu"], composition = {"Pt":2,"Fe":3} corresponds to Pt2Fe3Cu

    lattice_parameters:
        Dictionary for lattice parameters <a> for each species.
        If not specified, defaults from the `ase.data` module are used

    crystal_structure:
        String indicated the crystal structure of the skeleton lattice to be populated.
        Defaults to fcc

    facets:
        List of surface facets to be considered when generating the randomly populated MPEA.

        The following defaults will be used based on the crystal structure unless
        otherwise specified:
        fcc/bcc: 100,111,110
        hcp: 0001

    supcell: 
        Tuple or List specifying the size of the supercell to be
        generated in the format (nx,ny,nz).

    num_of_samples:
        Integer specifying the number of samples to be taken for each surface facet
        (e.g. num_of_samples = 15 will generate 15 structures for each facet)

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

    mpeas:
        Dictionary containing all generated mpeas for each surface facet
        (and corresponding paths to the structures if written to disk)

    """

    if composition is None:
        composition = {}

    comp_library = {species: 1.0 for species in species_list}
    comp_library.update(composition)

    # Get MPEA name from species list and composition
    name = ""
    for species in species_list:
        name += species + str(comp_library[species])

    ft_defaults = {
        "fcc": ["100", "111", "110"],
        "bcc": ["100", "111", "110"],
    }

    if facets is None:
        facets = ft_defaults[crystal_structure]

    mpeas = {}
    for ft in facets:
        mpeas[crystal_structure + ft] = {}
        for i in range(num_of_samples):
            samp = random_population(
                species_list=species_list,
                composition=composition,
                lattice_parameters=lattice_parameters,
                crystal_structure=crystal_structure,
                ft=ft,
                supcell=supcell,
                vac=vac,
                fix=fix,
            )
            traj_file_path = None
            if write_to_disk:
                dir_path = os.path.join(
                    write_location, name + "/" + crystal_structure + ft + "/" + str(i)
                )
                os.makedirs(dir_path, exist_ok=dirs_exist_ok)
                traj_file_path = os.path.join(dir_path, "input.traj")
                samp.write(traj_file_path)
                print(
                    f"Sample #{str(i)} for {crystal_structure + ft} written to {traj_file_path}"
                )

            mpeas[crystal_structure + ft][str(i)] = {
                "structure": samp,
                "traj_file_path": traj_file_path,
            }
    return mpeas


def random_population(
    species_list: List[str],
    composition: Dict[str, float] = None,
    lattice_parameters: Dict[str, float] = None,
    crystal_structure: str = "fcc",
    ft: str = "100",
    supcell: Union[Tuple[int], List[int]] = (3, 3, 4),
    vac: float = 10.0,
    fix: int = 0,
):
    """
    Returns a randomly populated structure from a species list and composition

    Parameters
    ----------

    species_list:
        List of species that will populate the skeleton structure

    composition:
        Dictionary of desired composition, defaults to be 1/sum(composition) for each species
        not mentioned. Used for calculating probabilities of each species occupying a given site.

        e.g. species_list = ["Pt","Fe","Cu"], composition = {"Pt":2,"Fe":3} corresponds to Pt2Fe3Cu

    lattice_parameters:
        Dictionary for lattice parameters <a> for each species.
        If not specified, defaults from the `ase.data` module are used

    crystal_structure:
        String indicated the crystal structure of the skeleton lattice to be populated.
        Defaults to fcc

    ft:
        String indicating the surface facet to be considered.
        Defaults to 100

    supcell:
        Tuple or List specifying the size of the supercell to be
        generated in the format (nx,ny,nz).

    Returns
    -------

    rand_struct:
        Atoms object for the randomly populated structure

    """

    if composition is None:
        composition = {}

    comp_library = {species: 1.0 for species in species_list}
    comp_library.update(composition)

    comp_list = list(comp_library.values())
    comp_sum = np.sum(comp_list)
    p = np.array(comp_list) / comp_sum

    if lattice_parameters is None:
        lattice_parameters = {}

    latt_library = {
        species: reference_states[atomic_numbers[species]].get("a")
        for species in species_list
    }

    latt_library.update(lattice_parameters)

    # calculate lattice parameter as weighted average based on composition
    a = np.average(list(latt_library.values()), weights=p)

    # Generate atoms list
    num_atoms = np.prod(supcell)
    atoms_list = []
    i = 0
    while i < num_atoms:
        atoms_list.append(np.random.choice(species_list, p=p))
        i += 1

    # use the first species to build a skeleton lattice which will be populated
    skel = species_list[0]
    rand_struct = generate_surface_structures(
        [skel],
        supcell=supcell,
        vac=vac,
        fix=fix,
        a_dict={skel: a},
        crystal_structures={skel: crystal_structure},
        ft_dict={skel: [ft]},
    )[skel][crystal_structure + ft].get("structure")

    rand_struct.set_chemical_symbols(atoms_list)

    return rand_struct
