import os
import numpy as np
from ase.io import read,write
from ase import Atom,Atoms
from ase.build import add_adsorbate
from ase.build import molecule
from ase.visualize import view
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


def gen_rxn_int_sym(surf, site_type=['ontop','bridge','hollow'], ads=['NNH','NH2'],height=[1.5,1.5],rots=[[[0.0,'x']],[[180.,'x']]]):
    '''
    
    Given site types will create subdirectories for each identified symmetry site of that type

    Parameters:
        surf(str or ase obj): name of traj file of relaxed surface to be adsorbed upon
        site_type(list of str): desired types of adsorption symmetry sites to identify
            Options: 'ontop', 'bridge', 'hollow'
        ads(list of str): list of adsorbates to be placed

    Returns:
        None

    '''
    curr_dir = os.getcwd()

    sites = get_ads_sites(surf, ads_site_type=site_type) # gets dict containing identified sites

    for typ in sites.keys():
        if typ != 'all':
            for p in sites[typ]:
                gen_rxn_int_pos(surf,ads=ads,pos=p[:2],height=height,rots=rots,label=typ)

    print('Completed')

def gen_rxn_int_pos(surf, ads=['NNH','NH2'],pos=(0.,0.),height=[1.5,1.5],rots=[[[0.0,'x']],[[180.,'x']]],label='custom'):
    '''

    Given relaxed structure & xy position, generates new directories containing traj files of each adsorbate placed over pos

    Parameters:
        surf(str or ase obj): name of traj file containing relaxed surface to be adsorbed upon
        ads(list of str): list of adsorbates to be placed
        pos(tuple,list,or np.array of floats): xy coordinate for where to place adsorbate
        height(list of float): height to place each adsorbate over surface
        rots(list of list of list of float and str): list of list of rotation operations for each adsorbate
        label(str): label for selected position (e.g. ontop, bridge, custom, etc.)

    Returns:
        None 

    '''

    #sa_ind = find_sa_ind(surf)
    #pos_x = saa[sa_ind].x
    #pos_y = saa[sa_ind].y

    rpos = np.around(pos,3)

    curr_dir = os.getcwd()
    i = 0
    while i < len(ads): # iterates over each of the given adsorbates
        try:
            os.makedirs(ads[i] + '/' + label + '/' + str(rpos[0]) + '_' + str(rpos[1]))
        except OSError:
            print('Failed Creating Directory ./{}'.format(ads[i]+ '/' + label + '/' + str(rpos[0]) + '_' + str(rpos[1])))
        else:
            print('Successfully Created Directory ./{}'.format(ads[i]+ '/' + label + '/' + str(rpos[0]) + '_' + str(rpos[1])))
            os.chdir(ads[i]+ '/' + label + '/' + str(rpos[0]) + '_' + str(rpos[1]))
            place_adsorbate(surf,ads[i],position=pos,height=height[i],write_traj=True,rotations=rots[i])
            print('Adsorbed {} traj generated at position {}'.format(ads[i],rpos))
            os.chdir(curr_dir)
        i += 1

    print('Completed traj generation for position {}'.format(rpos))

def find_sa_ind(saa):
    ''' Given an SAA structure, finds the single atom index (and cartesian xy)'''
    syms = np.array(saa.symbols)
    unique, counts = np.unique(syms, return_counts=True)
    ind = np.where(syms==unique[np.argmin(counts)])[0][0] # Finds index of species with lowest count
    return ind


def place_adsorbate(surface, mol, position = (0.0,0.0), height = 1.5, rotations=[[0.0,'x']], write_traj=False):
    '''
    Parameters:
        surface(str or ase obj): surface to adsorb onto. Either str of filename to be read or ase obj
        mol(str or ase obj): name of molecule to be adsorbed or ase atoms obj
        position(tuple of floats): cartesian coordinates of where the molecule should be placed in the xy plane
        height(float): height above surface where adsorbate should be initially placed
        rotation(tuple of floats): rotation of molecule to be applied about the x,y,z axes
        write_traj(bool): whether to write out traj file

    Returns:
        ads_state(ase obj): surface with the adsorbant placed

    '''

    # Identify if surface specified as filename or atoms object
    if type(surface) is str:
        surf = read(surface)

    elif type(surface) is Atoms:
        surf = surface.copy()

    else:
        raise TypeError('surface parameter needs to be either a str or ase.Atoms object')


    # Identify if molecule is specified by name or atoms object
    if type(mol) is str:
        adsorbate = generate_molecule_object(mol, rotations = rotations)
    elif type(mol) is Atoms:
        adsorbate = mol.copy()
        for r in rotations:
            adsorbate.rotate(r[0],r[1])

    lowest_ind = adsorbate.positions[:,2].argmin()

    add_adsorbate(surf, adsorbate, height, position = position,mol_index=lowest_ind)

    if write_traj:
        surf.write(mol + '.i.traj')


    return surf




def generate_molecule_object(mol, rotations = [[0.0,'x']]):
    '''
    Parameters:
        mol(str): name of molecule to be generated
        rotations(list of floats and strings): list of rotation operations to be carried out

    Returns:
        m(ase Atoms obj): molecule object
    '''

    if mol in g2.names:
        m = molecule(mol)
        for r in rotations:
            m.rotate(r[0],r[1])
        lowest_mol = np.min(m.positions[:,2])
        for atom in m:
            atom.position[2] -= lowest_mol
        return m

    elif mol == 'N2H' or mol == 'NNH':
        m = Atoms('N2H',[(0.,0.,0.,),(0.,0.,1.2),(0.71,0,1.91)])
        for r in rotations:
            m.rotate(r[0],r[1])
        return m

    elif mol == 'N2Hh' or mol == 'NNHh':
        m = Atoms('N2H',[(-0.55,0.,0.),(0.55,0.,0.),(0.6,0.,1.)])
        for r in rotations:
            m.rotate(r[0],r[1])
        return m

    return None


def get_ads_sites(surface, ads_site_type=['ontop','bridge','hollow']):
    '''
    
    From name of file to be read or aseobj, returns dictionary of adsorption sites with the keys given in ads_site_type
    
    Parameters:
        surface(str or ase obj): surface to find sites for
        ads_site_type(list of str): types of sites to look for (options are 'ontop','bridge', and 'hollow')

    Returns:
        sites(dict of list of np.arrays): dict with keys being ads_site_type giving a list of the identified sites
    
    '''
    # Reads in surface if necessary
    if type(surface) is str:
        surf = read(surface)
    
    elif type(surface) is Atoms:
        surf = surface.copy()

    else:
        raise TypeError('surface parameter needs to be either a str or ase.Atoms object')


    conv = AseAtomsAdaptor() # define convertor from ase object to pymatgen structure
    
    struct = conv.get_structure(surf) # make conversion to mg structure object
    
    finder = AdsorbateSiteFinder(struct) # define site finder
    
    sites = finder.find_adsorption_sites(positions=ads_site_type)
    
    return sites


def view_ads_sites(surface,ads_site_type=['ontop','bridge','hollow'],save_im=False,view_im=True,supcell=(1,1,1)):
    '''
    From given surface, visualizes each of the ads_site_type specified.
    
    Parameters
        surface(str or ase obj): Surface to find sites for 
        save_im(bool): True if the images are to be saved
        view_im(bool): True if the sites are to be viewed using the ase-gui
        supcell(tuple of ints): supercell size for visualization purposes

    Returns:
        None
    '''
    # Gets the adsorption sites
    sites = get_ads_sites(surface, ads_site_type)

    # Reads in surface if necessary
    if type(surface) is str:
        ase_obj = read(surface)

    elif type(surface) is Atoms:
        ase_obj = surface.copy()

    else:
        raise TypeError('surface parameter needs to be either a str or ase.Atoms object')

    name = ase_obj.get_chemical_formula()
    for ads_type in ads_site_type: # Iterates over site type
        ase_obj_i = ase_obj.copy()
        for site in sites[ads_type]: # Iterates over site given type
            ase_obj_i.append(Atom('X',position=site)) # Adds placeholder atom at site
        if save_im:
            write(str(name)+'_'+str(ads_type)+'_sites.png',ase_obj_i*supcell) # Saves image
        if view_im: # Visualizes each site type
            view(ase_obj_i*supcell)



#def get_ads_struct(sub_file, mol, write_traj=False):
#    '''Given name of substrate file or ase obj, adsorbs mol (str) onto each of the identified adsorption sites.'''
#
#    try:
#        sub_ase = read(sub_file)
#    except AttributeError:
#        sub_ase = sub_file
#
#    tags = sub_ase.get_tags()
#
#    conv = AseAtomsAdaptor() # converter between pymatgen and ase
#    
#    struct = conv.get_structure(sub_ase) # convert ase substrate to pymatgen structure
#
#    finder = AdsorbateSiteFinder(struct)
#
#    m_ase = Atoms(mol) # create ase atoms object of desired adsorbate
#    
#    molecule = conv.get_molecule(m_ase) # convert adsorbate to pymatgen molecule
#
#    all_structs = finder.generate_adsorption_structures(molecule) # collect all adsorption structures
#
#    
#    all_ase_structs = []
#    i = 0
#    while i < len(all_structs):
#        ase_struct = conv.get_atoms(all_structs[i]) # converts to ase object
#        #ase_struct.set_tags(tags)
#        all_ase_structs.append(ase_struct)
#        if write_traj:
#            write('struct_'+str(i)+'.traj',ase_struct) # writes all adsorption structures to a separate ase traj file
#        i += 1
#
#
#    return all_ase_structs



if __name__ == '__main__':
    saa = read('CuFe0.traj')
    #gen_rxn_int_pos(saa, ads=['NNH','NH2'],pos=np.array([0.,0.]),height=[1.5,1.5],rots=[[[0.0,'x']],[[180.,'x']]])
    gen_rxn_int_sym(saa, site_type=['ontop'])
    #s = read('clean/Pt16.traj')
    #gen_rxn_int_sym(s, site_type=['ontop','hollow'],ads=['CO'],height=[1.5],rots=[[[0.0,'x']]])
