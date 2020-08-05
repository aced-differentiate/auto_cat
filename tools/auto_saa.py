import os
import numpy as np
from ase.io import read,write
from ase import Atom,Atoms
from ase.visualize import view
from ase.build import fcc100,fcc110,fcc111
from ase.build import bcc100,bcc110,bcc111
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


def gen_saa(subs, dops, bv = 'fcc',ft=['100','110','111'],supcell=(3,3,4),a=None,cent_sa=True):
    '''
    For each of the substrates and dopants specified, a new directory containing a traj file for the SAA with the 
    specified surfaces is generated

    Parameters:
        subs (list of str): host species
        dops (list of str): dopant species
        bv (str): bravais lattice (currently only fcc and bcc implemented)
        ft (list of str): facets to be considered
        supcell (tuple): supercell

    Returns:
        None
    '''
    i = 0
    while i < len(subs):
        hosts = gen_hosts([subs[i]],bv,ft,supcell,a) # generate host structures
        j = 0
        while j < len(dops): # iterate over dopants
            for f in ft: # iterate over facets
                try:
                    os.mkdir(subs[i]+'_'+dops[j]+'_'+bv+f) # create directory for each sub/dop combo
                except OSError:
                    print('Failed Creating Directory ./{}_{}_{}'.format(subs[i],dops[j],bv+f))
                else:
                    print('Successfully Created Directory ./{}_{}_{}'.format(subs[i],dops[j],bv+f))
                    os.chdir(subs[i]+'_'+dops[j]+'_'+bv+f) # change into new dir
                    slab = hosts[bv+f] # extract host corresponding to facet
                    gen_doped_structs(slab,dops[j],write_traj=True,cent_sa=cent_sa) #  generate doped structures
                    print('{}_{}_{} SAA trajs generated'.format(subs[i],dops[j],bv+f))
                    os.chdir('..')
            j += 1
        i += 1

    print('Completed')



def gen_doped_structs(sub_ase,dop, write_traj=False, cent_sa = True):
    '''
    Generates doped structures given host material (ase object) and dopant (str), and returns list of ase objs

    Parameters:
        sub_ase (ase Atoms obj): host material
        dop (str): dopant species
        cen_saa (bool): If centering of SA in the cell is desired, otherwise will be placed at the origin

    Return
        all_ase_structs (list of ase Atoms obj): doped structures

    '''
    name = "".join(np.unique(sub_ase.symbols))
    tags = sub_ase.get_tags()
    conv = AseAtomsAdaptor() # converter between pymatgen and ase

    struct = conv.get_structure(sub_ase) # convert ase substrate to pymatgen structure

    finder = AdsorbateSiteFinder(struct)


    all_structs = finder.generate_substitution_structures(dop) # collect all adsorption structures

    all_ase_structs= []
    i = 0
    while i < len(all_structs):
        ase_struct = conv.get_atoms(all_structs[i]) # converts to ase object
        ase_struct.set_tags(tags)
        if cent_sa: # centers the sa
            cent_x = ase_struct.cell[0][0]/2 + ase_struct.cell[1][0]/2
            cent_y = ase_struct.cell[0][1]/2 + ase_struct.cell[1][1]/2
            cent = (cent_x,cent_y,0)
            ase_struct.translate(cent)
            ase_struct.wrap()

        all_ase_structs.append(ase_struct)
        if write_traj:
            write(name+dop+str(i)+'.traj',ase_struct)
        i += 1

    return all_ase_structs


def gen_hosts(species, bv = 'fcc', ft=['100','110','111'],supcell=(3,3,4),a=None):
    '''
    Given list of host species, bravais lattice, and facets, generates dict of ase objects for hosts

    Parameters
    species (list of str): list of host species
    bv (str): bravais lattice
    ft (list of str): facets to be considered
    supcell (tuple): supercell size

    Returns
    hosts (dict): dictionary of generated host materials
    '''
    hosts = {}
    for s in species:
        funcs = {'fcc100':fcc100,'fcc110':fcc110,'fcc111':fcc111,'bcc100':bcc100,'bcc110':bcc110,'bcc111':bcc111}
        j =0
        while j < len(ft):
            if a is None:
                hosts[bv+ft[j]]=funcs[bv+ft[j]](str(s),size=supcell,vacuum=10.0)
            else:
                hosts[bv+ft[j]]=funcs[bv+ft[j]](str(s),size=supcell,vacuum=10.0,a=a)
            j += 1
    return hosts
            

def dope_surface(surf, sites, dop, write_traj=False):
    '''
    Given a surface (aseobj) and dopant species(str), returns a list of ase object of doped surfaces.
    NOTE: Allows for MANUAL selection of substition site
    Optionally writes traj files for each struct

    Parameters
    surf (ase Atoms object): host object
    sites (list of ints): list of atom indices to be substituted

    Returns
    doped_surfs (list of ase Atoms objects): doped structures
    '''
    surf_name = surf.get_chemical_symbols()[0]
    doped_surfs = []
    for site in sites: # iterate over all given top sites
        dop_surf = surf.copy()
        dop_surf[site].symbol = dop # updates atom at top site to dopant
        doped_surfs.append(dop_surf)

        if write_traj:
            i = 0
            while i < len(doped_surfs):
                doped_surfs[i].write(surf_name+'_'+dop+'_'+str(site)+'.traj') # writes the traj files to the directory
                i += 1
    return doped_surfs


#def top_pos_to_ind(surf,sites):
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


if __name__ == '__main__':
    gen_saa(['Cu'],['Fe'],ft=['111'],supcell=(3,3,4),a=1.813471*2)
    #gen_saa(['Cu'],['Fe'],supcell=(5,5,4),a=1.813471*2)
