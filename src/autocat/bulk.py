import os
import numpy as np
from ase.io import read, write
from ase import Atom, Atoms
from ase.visualize import view
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from autocat.adsorption import place_adsorbate, get_ads_sites


def gen_bulk(species, bv="fcc", ft=["100", "110", "111"], supcell=(3, 3, 4), a=None):
    """
    Given bulk species, bravais lattice, and facets, generates dict of ase objects for bulk

    Parameters
    species (str): bulk species
    bv (str): bravais lattice
    ft (list of str): facets to be considered
    supcell (tuple): supercell size

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
    return bulk
