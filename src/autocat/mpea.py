import os
import numpy as np
from ase.io import read, write
from ase import Atom, Atoms
from ase.visualize import view
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


def gen_mpea(
    spec_list, comp, bv="fcc", ft=["100", "110", "111"], supcell=(3, 3, 4), samps=15
):
    """

    For the given species list and composition will generate a specified number of samples from this space
    in separate directories

    Parameters:
        spec_list(list of str): names of chemical species to be included
        comp(list of floats): composition of species list
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
    while j < len(spec_list):
        name += spec_list[j]
        name += str(comp[j])
        j += 1

    curr_dir = os.getcwd()

    for f in ft:
        try:
            os.mkdir(name + "_" + bv + f)
        except OSError:
            print("Failed Creating Directory ./{}".format(name + "_" + bv + f))
        else:
            print("Successfully Created Directory ./{}".format(name + "_" + bv + f))
            os.chdir(name + "_" + bv + f)
            sub_dir = os.getcwd()
            print("Beginning Generation of MPEAS for facet {}{}".format(bv, f))
            i = 0
            while i < samps:
                os.mkdir(str(i + 1))
                os.chdir(str(i + 1))
                gen_mpea_struct(
                    spec_list, comp, bv=bv, ft=f, supcell=supcell, write_traj=True
                )
                os.chdir(sub_dir)
                i += 1
            os.chdir(curr_dir)
            print("Completed traj generation for facet {}{}".format(bv, f))
    print("Successfully completed traj generation for {}".format(name + "_" + bv + f))

    return None


def gen_mpea_struct(
    spec_list, comp, bv="fcc", ft="100", supcell=(3, 3, 4), write_traj=False
):
    """
    Returns a random mpea traj structure given a species list and composition

    Parameters:
        spec_list(list of str): list of species to include
        comp(list of floats): desired composition
        bv(str): bravais lattice of structure (currently only supports fcc & bcc)
        ft(str): facet to be considered
        supcell(tuple of ints): supercell size
        write_traj(bool): whether to write the traj file or not

    Returns:
        mpea(ase obj): randomly generated mpea

    """
    comp_sum = np.sum(comp)
    p = np.array(comp) / comp_sum

    # Optimimum lattice parameters calculated using BEEF-vdW
    latt_beef = {"Pt": 2.014179 * 2, "Ir": 3.84}  # Need to fix Ir

    # Average lattice parameter given composition
    a = 0
    i = 0
    while i < len(spec_list):
        a += latt_beef[spec_list[i]] * p[i]
        i += 1

    # Generate atoms list
    num_atoms = np.prod(supcell)

    atoms_list = []
    i = 0
    while i < num_atoms:
        atoms_list.append(np.random.choice(spec_list, p=p))
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

    host = funcs[bv + ft](spec_list[0], size=supcell, vacuum=10.0, a=a)

    host.set_chemical_symbols(atoms_list)

    if write_traj:
        name = ""
        j = 0
        while j < len(spec_list):
            name += spec_list[j]
            name += str(comp[j])
            j += 1
        host.write(name + ".i.traj")
    return host
