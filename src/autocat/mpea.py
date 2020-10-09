import os
import numpy as np
from ase.io import read, write
from ase import Atom, Atoms
from ase.visualize import view
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from ase.data import reference_states, atomic_numbers
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


def gen_mpea_dirs(
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


def gen_mpea(
    species_list,
    comp={},
    latts={},
    bv="fcc",
    ft="100",
    supcell=(3, 3, 4),
    write_traj=False,
):
    """
    Returns a random mpea traj structure given a species list and composition

    Parameters:
        species_list(list of str): list of species to include
        comp(dict): desired composition, if species in list not mentioned, assumed to be 1/sum(comp)
        latts(dict): lattice parameters for each species (defaults to ASE values if not specified)
        bv(str): bravais lattice of structure (currently only supports fcc & bcc)
        ft(str): facet to be considered
        supcell(tuple of ints): supercell size
        write_traj(bool): whether to write the traj file or not

    Returns:
        mpea(ase obj): randomly generated mpea

    """

    # Checks if any of the species in the species list are specified in comp
    # Otherwise sets it to 1

    i = 0
    comp_list = []
    while i < len(species_list):
        if species_list[i] not in comp:
            comp_list.append(1.0)
        else:
            comp_list.append(comp[species_list[i]])
        i += 1

    comp_sum = np.sum(comp_list)
    p = np.array(comp_list) / comp_sum

    # Average lattice parameter given composition
    a = 0
    i = 0
    while i < len(species_list):
        if species_list[i] in latts:
            a += latts[species_list[i]] * p[i]
        else:
            a += reference_states[atomic_numbers[species_list[i]]]["a"] * p[i]
        i += 1

    # Generate atoms list
    num_atoms = np.prod(supcell)

    atoms_list = []
    i = 0
    while i < num_atoms:
        atoms_list.append(np.random.choice(species_list, p=p))
        i += 1

    # Build structure

    funcs = {
        "fcc100": fcc100,
        "fcc110": fcc110,
        "fcc111": fcc111,
        "bcc100": bcc100,
        "bcc110": bcc110,
        "bcc111": bcc111,
    }

    host = funcs[bv + ft](species_list[0], size=supcell, vacuum=10.0, a=a)

    host.set_chemical_symbols(atoms_list)

    if write_traj:
        name = ""
        j = 0
        while j < len(species_list):
            name += species_list[j]
            name += str(comp_list[j])
            j += 1
        host.write(name + ".i.traj")
    return host
