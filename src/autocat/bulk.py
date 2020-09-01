import os
from ase.io import read, write
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from ase.constraints import FixAtoms


def gen_bulk_dirs(
    species_list,
    bv=["fcc"],
    ft=["100", "110", "111"],
    supcell=(3, 3, 4),
    a=[None],
    fix=0,
):
    """
    Given list of species, bravais lattice, and facets creates directories containing traj files for the surfaces

    Parameters:
        species_list(list of str): list of bulk species to be generated
        bv(list of str): list of bravais lattices corresponding to each species
        ft(list of str): list of facets to consider (should be same length as species list)
        supcell(tuple of int): supercell size to be generated
        a(list of float): lattice parameters for each species (same length as species list). if None then uses exp. value

    Returns:
        None
    """
    curr_dir = os.getcwd()
    i = 0
    while i < len(species_list):
        b = gen_bulk(species_list[i], bv=bv[i], ft=ft, supcell=supcell, a=a[i], fix=fix)
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


def gen_bulk(
    species,
    bv="fcc",
    ft=["100", "110", "111"],
    supcell=(3, 3, 4),
    a=None,
    fix=0,
    write_traj=False,
):
    """
    Given species, bravais lattice, and facets, generates dict of ase objects for surfaces

    Parameters
    species (str): bulk species
    bv (str): bravais lattice
    ft (list of str): facets to be considered
    supcell (tuple): supercell size
    a (float): lattice parameter. if None uses experimental value
    fix (int): number of layers from bottom to fix (e.g. value of 2 fixes bottom 2 layers)

    Returns
    bulk (dict): dictionary of generated bulk facets
    """
    bulk = {}
    funcs = {
        "fcc100": fcc100,
        "fcc110": fcc110,
        "fcc111": fcc111,
        "bcc100": bcc100,
        "bcc110": bcc110,
        "bcc111": bcc111,
    }
    j = 0
    while j < len(ft):
        if a is None:
            bulk[bv + ft[j]] = funcs[bv + ft[j]](species, size=supcell, vacuum=10.0)
        else:
            bulk[bv + ft[j]] = funcs[bv + ft[j]](
                species, size=supcell, vacuum=10.0, a=a
            )
        j += 1
    if fix > 0:
        for sys in bulk:
            f = FixAtoms(mask=[atom.tag > (supcell[-1] - fix) for atom in bulk[sys]])
            bulk[sys].set_constraint([f])

    if write_traj:
        for sys in bulk:
            bulk[sys].write(species + "_" + sys + ".i.traj")
    return bulk
