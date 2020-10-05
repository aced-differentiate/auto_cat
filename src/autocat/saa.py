import os
import numpy as np
from ase.io import read, write
from ase import Atom, Atoms
from ase.visualize import view
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from ase.data import atomic_numbers, ground_state_magnetic_moments, reference_states
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from autocat.surface import gen_surf


def gen_saa_dirs(
    subs,
    dops,
    bv_dict={},
    ft=["100", "110", "111"],
    supcell=(3, 3, 4),
    a_dict={},
    c_dict={},
    cent_sa=True,
    fix=0,
):
    """
    For each of the substrates and dopants specified, a new directory containing a traj file for the SAA with the 
    specified surfaces is generated

    Parameters:
        subs (list of str): host species
        dops (list of str): dopant species
        bv_dict (dict): dict of manually specified bravais lattices for specific host species
        ft (list of str): facets to be considered
        a_dict(dict): manually specified lattice parameters for species. if None then uses ASE default
        c_dict(dict): manually specified lattice parameters for species. if None then uses ASE default
        supcell (tuple): supercell
        fix (int): number of layers from bottom to fix (e.g. value of 2 fixes bottom 2 layers)

    Returns:
        None
    """

    curr_dir = os.getcwd()

    i = 0
    while i < len(subs):

        # Check if Bravais lattice manually specified
        if subs[i] in bv_dict:
            bv = bv_dict[subs[i]]
        else:
            bv = reference_states[atomic_numbers[subs[i]]]["symmetry"]

        # Check if a manually specified
        if subs[i] in a_dict:
            a = a_dict[subs[i]]
        else:
            a = None

        # Check if c manually specified
        if subs[i] in c_dict:
            c = c_dict[subs[i]]
        else:
            c = None

        hosts = gen_surf(
            subs[i], bv=bv, ft=ft, supcell=supcell, a=a, c=c, fix=fix
        )  # generate host structures
        j = 0
        while j < len(dops):  # iterate over dopants
            if subs[i] != dops[j]:  # ensures different host and sa species
                for f in ft:  # iterate over facets
                    try:
                        os.makedirs(
                            subs[i] + "/" + dops[j] + "/" + bv + f
                        )  # create directory for each sub/dop combo
                    except OSError:
                        print(
                            "Failed Creating Directory ./{}/{}/{}".format(
                                subs[i], dops[j], bv + f
                            )
                        )
                    else:
                        print(
                            "Successfully Created Directory ./{}/{}/{}".format(
                                subs[i], dops[j], bv + f
                            )
                        )
                        os.chdir(
                            subs[i] + "/" + dops[j] + "/" + bv + f
                        )  # change into new dir
                        slab = hosts[bv + f]  # extract host corresponding to facet
                        gen_doped_structs(
                            slab, dops[j], write_traj=True, cent_sa=cent_sa
                        )  #  generate doped structures
                        print(
                            "{}/{}/{} SAA trajs generated".format(
                                subs[i], dops[j], bv + f
                            )
                        )
                        os.chdir(curr_dir)
            j += 1
        i += 1

    print("Completed")


def gen_doped_structs(sub_ase, dop, write_traj=False, cent_sa=True):
    """
    Generates doped structures given host material (ase object) and dopant (str), and returns list of ase objs

    Parameters:
        sub_ase (ase Atoms obj): host material
        dop (str): dopant species
        cen_saa (bool): If centering of SA in the cell is desired, otherwise will be placed at the origin

    Return
        all_ase_structs (list of ase Atoms obj): doped structures

    """
    name = "".join(np.unique(sub_ase.symbols))
    tags = sub_ase.get_tags()
    constr = sub_ase.constraints
    conv = AseAtomsAdaptor()  # converter between pymatgen and ase

    struct = conv.get_structure(sub_ase)  # convert ase substrate to pymatgen structure

    finder = AdsorbateSiteFinder(struct)

    all_structs = finder.generate_substitution_structures(
        dop
    )  # collect all substitution structures

    # magmom guess for dopant. Taken from http://www.webelements.com
    mag = ground_state_magnetic_moments[atomic_numbers[dop]]

    all_ase_structs = []
    i = 0
    while i < len(all_structs):
        ase_struct = conv.get_atoms(all_structs[i])  # converts to ase object
        ase_struct.set_tags(tags)
        ase_struct.pbc = (1, 1, 0)  # ensure pbc in xy only
        ase_struct.constraints = constr  # propagate constraints
        sa_ind = find_sa_ind(ase_struct)
        ase_struct[sa_ind].magmom = mag  # set initial magmom
        if cent_sa:  # centers the sa
            cent_x = ase_struct.cell[0][0] / 2 + ase_struct.cell[1][0] / 2
            cent_y = ase_struct.cell[0][1] / 2 + ase_struct.cell[1][1] / 2
            cent = (cent_x, cent_y, 0)
            ase_struct.translate(cent)
            ase_struct.wrap()

        all_ase_structs.append(ase_struct)
        if write_traj:
            write(name + dop + str(i) + ".i.traj", ase_struct)
        i += 1

    return all_ase_structs


def find_sa_ind(saa):
    """ Given an SAA structure (without any adsorbates), finds the single atom index"""
    syms = np.array(saa.symbols)
    unique, counts = np.unique(syms, return_counts=True)
    ind = np.where(syms == unique[np.argmin(counts)])[0][
        0
    ]  # Finds index of species with lowest count
    return ind


def dope_surface(surf, sites, dop, write_traj=False):
    """
    Given a surface (aseobj) and dopant species(str), returns a list of ase object of doped surfaces.
    NOTE: Allows for MANUAL selection of substition site
    Optionally writes traj files for each struct

    Parameters
    surf (ase Atoms object): host object
    sites (list of ints): list of atom indices to be substituted

    Returns
    doped_surfs (list of ase Atoms objects): doped structures
    """
    surf_name = surf.get_chemical_symbols()[0]
    doped_surfs = []

    # magmom guess for dopant. Taken from http://www.webelements.com
    mag = ground_state_magnetic_moments[atomic_numbers[dop]]

    for site in sites:  # iterate over all given top sites
        dop_surf = surf.copy()
        dop_surf[site].symbol = dop  # updates atom at top site to dopant
        dop_surf[site].magmom = mag  # guesses initial magnetic moment
        doped_surfs.append(dop_surf)

        if write_traj:
            i = 0
            while i < len(doped_surfs):
                doped_surfs[i].write(
                    surf_name + "_" + dop + "_" + str(site) + ".i.traj"
                )  # writes the traj files to the directory
                i += 1
    return doped_surfs


# def top_pos_to_ind(surf,sites):
#    '''Takes top site positions from get_ads_sites and converts them to corresponding atom indices of the surface (aseobj)'''
#    inds = []
#    for site in sites: # iterates over all site positions
#        i = 0
#        while i < len(surf): # iterate over all atoms in the cell
#            # checks if the site position is equal to the i-th atom in the x-y coordinates
#            if surf.positions[i][0] == site[0] and surf.positions[i][1] == site[1]:
#                inds.append(i)
#            i += 1
#    return inds
