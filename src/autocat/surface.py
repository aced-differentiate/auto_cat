import os
from ase.io import read, write
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from ase.build import hcp0001
from ase.data import reference_states, atomic_numbers
from ase.constraints import FixAtoms


def gen_surf_dirs(
    species_list,
    bv_dict={},
    ft=["100", "110", "111"],
    supcell=(3, 3, 4),
    a_dict={},
    c_dict={},
    fix=0,
):
    """
    Given list of species, bravais lattice, and facets creates directories containing traj files for the surfaces

    Parameters:
        species_list(list of str): list of surf species to be generated
        bv(dict): dict of manually specified bravais lattices for specific species
        ft(list of str): list of facets to consider (should be same length as species list)
        supcell(tuple of int): supercell size to be generated
        a(dict): manually specified lattice parameters for species. if None then uses ASE default
        c(dict): manually specified lattice parameters for species. if None then uses ASE default

    Returns:
        None
    """
    curr_dir = os.getcwd()
    i = 0
    while i < len(species_list):

        # Check if Bravais lattice manually specified
        if species_list[i] in bv_dict:
            bv = bv_dict[species_list[i]]
        else:
            bv = None

        # Check if a manually specified
        if species_list[i] in a_dict:
            a = a_dict[species_list[i]]
        else:
            a = None

        # Check if c manually specified
        if species_list[i] in c_dict:
            c = c_dict[species_list[i]]
        else:
            c = None

        b = gen_surf(species_list[i], bv=bv, ft=ft, supcell=supcell, a=a, c=c, fix=fix)
        for facet in b.keys():
            try:
                os.makedirs(species_list[i] + "/" + facet)
            except OSError:
                print(
                    "Failed Creating Directory ./{}".format(
                        species_list[i] + "/" + facet
                    )
                )
            else:
                print(
                    "Successfully Created Directory ./{}".format(
                        species_list[i] + "/" + facet
                    )
                )
                os.chdir(species_list[i] + "/" + facet)
                b[facet].write(species_list[i] + "_" + facet + ".i.traj")
                os.chdir(curr_dir)
        i += 1
    print("Completed")


def gen_surf(
    species,
    bv=None,
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
    bv (str): bravais lattice
    ft (list of str): facets to be considered
    supcell (tuple): supercell size
    a (float): lattice parameter. if None uses ASE default
    c (float): lattice parameter. if None uses ASE default
    fix (int): number of layers from bottom to fix (e.g. value of 2 fixes bottom 2 layers)

    Returns
    surf (dict): dictionary of generated surf facets
    """

    if bv is None:  # uses ASE data to get Bravais Lattice
        bv = reference_states[atomic_numbers[species]]["symmetry"]

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
        surf[bv + ft[j]] = funcs[bv + ft[j]](
            species, size=supcell, vacuum=10.0, a=a, c=c
        )
        j += 1
    if fix > 0:
        for sys in surf:
            f = FixAtoms(mask=[atom.tag > (supcell[-1] - fix) for atom in surf[sys]])
            surf[sys].set_constraint([f])

    if write_traj:
        for sys in surf:
            surf[sys].write(species + "_" + sys + ".i.traj")
    return surf
