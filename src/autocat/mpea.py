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
    species_list,
    comp={},
    latts={},
    bv="fcc",
    ft=["100", "110", "111"],
    supcell=(3, 3, 4),
    samps=15,
):
    """

    For the given species list and composition will generate a specified number of samples from this space
    in separate directories

    Parameters:
        species_list(list of str): names of chemical species to be included
        comp(dict): desired composition, if species in list not mentioned, assumed to be 1/sum(comp)
        latts(dict): lattice parameters for each species (defaults to ASE values if not specified)
        bv(str): bravais lattice
        ft(list of str): facets to be considered
        supcell(tuple of int): supercell size
        samps(int): number of samples to be taken from this phase space

    Return:
        None
    """
    # Get MPEA name from species list and composition
    name = ""
    j = 0
    while j < len(species_list):
        name += species_list[j]
        if species_list[j] in comp:
            name += str(comp[species_list[j]])
        else:
            name += str(1)
        j += 1

    curr_dir = os.getcwd()

    for f in ft:
        try:
            os.makedirs(name + "/" + bv + f)
        except OSError:
            print("Failed Creating Directory ./{}".format(name + "/" + bv + f))
        else:
            print("Successfully Created Directory ./{}".format(name + "/" + bv + f))
            os.chdir(name + "/" + bv + f)
            sub_dir = os.getcwd()
            print("Beginning Generation of MPEAS for facet {}{}".format(bv, f))
            i = 0
            while i < samps:
                os.mkdir(str(i + 1))
                os.chdir(str(i + 1))
                gen_mpea(
                    species_list=species_list,
                    comp=comp,
                    latts=latts,
                    bv=bv,
                    ft=f,
                    supcell=supcell,
                    write_traj=True,
                )
                os.chdir(sub_dir)
                i += 1
            os.chdir(curr_dir)
            print("Completed traj generation for facet {}{}".format(bv, f))
    print("Successfully completed traj generation for {}".format(name + "_" + bv + f))

    return None


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
        not mentioned.

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

    # Checks if any of the species in the species list are specified in composition
    # Otherwise sets it to 1

    if composition is None:
        composition = {species: 1.0 for species in species_list}

    if lattice_parameters is None:
        lattice_parameters = {}

    comp_list = list(composition.values())
    comp_sum = np.sum(comp_list)
    p = np.array(comp_list) / comp_sum

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
