import os
import numpy as np
from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import Union

from ase.io import read, write
from ase import Atom, Atoms
from ase.build import add_adsorbate
from ase.build import molecule
from ase.visualize import view
from ase.data import chemical_symbols
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from autocat.intermediates.nrr import nrr_intermediate_names, nrr_mols
from autocat.intermediates.orr import orr_intermediate_names, orr_mols


def generate_rxn_structures(
    surf: Union[str, Atoms],
    site_type: List[str] = None,
    ads: List[str] = None,
    height: Dict[str, float] = None,
    rots: Dict[str, List[List[Union[float, str]]]] = None,
    site_im: bool = True,
    refs: List[Union[str, Atoms]] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    
    Given site types will create subdirectories for each identified symmetry site of that type

    Parameters:
        surf(str or ase Atoms obj): name of traj file of relaxed surface to be adsorbed upon
        site_type(list of str): desired types of adsorption symmetry sites to identify
            Options: 'ontop', 'bridge', 'hollow'
        ads(list of str or ase Atoms obj): list of adsorbates to be placed
        height(dict of float): height to place each adsorbate over surface
        rots(dict of list of list of float and str): dict of list of rotation operations for each specified adsorbate
            defaults to no applied rotations
        site_im(bool): writes out a traj showing all identified sites in a single file
        refs(list of str): names of reference states to be generated

    Returns:
        None

    """
    curr_dir = os.getcwd()

    sites = get_adsorption_sites(
        surf, ads_site_type=site_type
    )  # gets dict containing identified sites

    # writes out traj showing all identified symmetry sites
    if site_im:
        view_ads_sites(surf, ads_site_type=site_type, write_traj=True, view_im=False)

    print("Started building adsorbed structures")
    for typ in sites.keys():
        if typ != "all":
            for p in sites[typ]:
                gen_rxn_int_pos(
                    surf, ads=ads, pos=p[:2], height=height, rots=rots, label=typ
                )

    if refs is not None:
        print("Started building reference states")
        gen_refs_dirs(refs)

    print("Completed")


def gen_rxn_int_pos(
    surf, ads=["H"], pos=(0.0, 0.0), height={}, rots={}, label="custom"
):
    """

    Given relaxed structure & xy position, generates new directories containing traj files of each adsorbate placed over pos

    Parameters:
        surf(str or ase Atoms obj): name of traj file containing relaxed surface to be adsorbed upon
        ads(list of str or ase Atoms obj): list of adsorbates to be placed
        pos(tuple,list,or np.array of floats): xy coordinate for where to place adsorbate
        height(dict of float): height to place each adsorbate over surface
        rots(dict of list of list of float and str): dict of list of rotation operations for each specified adsorbate
            defaults to no applied rotations
        label(str): label for selected position (e.g. ontop, bridge, custom, etc.)

    Returns:
        None 

    """

    # sa_ind = find_sa_ind(surf)
    # pos_x = saa[sa_ind].x
    # pos_y = saa[sa_ind].y

    rpos = np.around(pos, 3)

    curr_dir = os.getcwd()
    i = 0
    while i < len(ads):  # iterates over each of the given adsorbates
        try:
            os.makedirs(ads[i] + "/" + label + "/" + str(rpos[0]) + "_" + str(rpos[1]))
        except OSError:
            print(
                "Failed Creating Directory ./{}".format(
                    ads[i] + "/" + label + "/" + str(rpos[0]) + "_" + str(rpos[1])
                )
            )
        else:
            print(
                "Successfully Created Directory ./{}".format(
                    ads[i] + "/" + label + "/" + str(rpos[0]) + "_" + str(rpos[1])
                )
            )
            os.chdir(ads[i] + "/" + label + "/" + str(rpos[0]) + "_" + str(rpos[1]))

            # checks if rotation for adsorbate specified
            if ads[i] in rots:
                r = rots[ads[i]]

            else:
                r = [[0.0, "x"]]

            # checks if height for adsorbate specified
            if ads[i] in height:
                h = height[ads[i]]

            else:
                h = 1.5  # defaults height 1.5 A

            place_adsorbate(
                surf, ads[i], position=pos, height=h, write_traj=True, rotations=r,
            )
            print("Adsorbed {} traj generated at position {}".format(ads[i], rpos))
            os.chdir(curr_dir)
        i += 1

    print("Completed traj generation for position {}".format(rpos))


def place_adsorbate(
    surface: Union[str, Atoms],
    mol: Union[str, Atoms],
    position: Union[Tuple[float], List[float]] = (0.0, 0.0),
    height: float = 1.5,
    rotations: List[List[Union[float, str]]] = None,
    mol_index: int = 0,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    Places an adsorbate onto a given surface. If specified will write
    the structure to disk

    Parameters
    ----------

    surface: 
        Atoms object or name of file containing structure 
        as a string specifying the surface for which the adsorbate should be placed

    mol:
        Atoms object or string of the name of the molecule to be generated. If the latter,
        will search in the `ase` g2 database first, then in `autocat.intermediates`

    position:
        Tuple or list of the xy cartesian coordinates for where the molecule should be placed

    height:
        Float specifying the height above surface where adsorbate should be initially placed

    rotations:
        List of rotation operations to be carried out which will be fed into
        the `ase.Atoms.rotate` method

        e.g. Rotating 90degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [[90.0,'z'],[45.0,'y']]

    mol_index:
        Integer index of atom in molecule that will be the reference for placing the molecule
        at (x,y). Will be fed into `ase.build.add_adsorbate` function.
        Defaults to the molecule at index 0

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

    surf:
        Atoms object of the surface with the molecule adsorbed

    """

    # Identify if surface specified as filename or atoms object
    if type(surface) is str:
        surf = read(surface)

    elif type(surface) is Atoms:
        surf = surface.copy()

    else:
        raise TypeError(
            "surface parameter needs to be either a str or ase.Atoms object"
        )

    if rotations is None:
        rotations = [[0.0, "x"]]

    # Identify if molecule is specified by name or atoms object
    if type(mol) is str:
        adsorbate = generate_molecule_object(mol, rotations=rotations)
        name = mol
    elif type(mol) is Atoms:
        adsorbate = mol.copy()
        name = mol.get_chemical_formula()
        for r in rotations:
            adsorbate.rotate(r[0], r[1])

    add_adsorbate(surf, adsorbate, height, position=position, mol_index=mol_index)

    if write_to_disk:
        rpos = np.around(position, 3)
        dir_path = os.path.join(
            write_location, name + "/" + str(rpos[0]) + "_" + str(rpos[1])
        )
        os.makedirs(dir_path, exist_ok=dirs_exist_ok)
        traj_file_path = os.path.join(dir_path, "input.traj")
        surf.write(traj_file_path)
        print(f"{name} structure written to {traj_file_path}")

    return surf


def gen_refs_dirs(refs_list, cell=[15, 15, 15]):
    """
    Parameters:
        refs_list(list of str): list of names of reference molecules to be generated
        cell: size of cell to contain molecule
    Returns:
        None
    """
    curr_dir = os.getcwd()

    for m in refs_list:
        try:
            os.makedirs("references/" + m)
        except OSError:
            print("Failed Creating Directory ./references/{}".format(m))
        else:
            print("Successfully Created Directory ./references/{}".format(m))
            os.chdir("references/" + m)
            generate_molecule_object(m, write_traj=True, cell=cell)
            print("Created {} reference state".format(m))
            os.chdir(curr_dir)
    print("Generation of reference states completed")


def generate_molecule_object(
    mol: str,
    rotations: List[List[Union[float, str]]] = None,
    cell: Union[List[int], Tuple[int]] = (15, 15, 15),
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    Generates an `ase.Atoms` object of an isolated molecule specified via a string
    If specified, can write out a traj containing the isolated molecule in a box

    Parameters
    ----------

    mol:
        String of the name of the molecule to be generated. Will search in the `ase` g2
        database first, then in `autocat.intermediates`

    rotations:
        List of rotation operations to be carried out which will be fed into
        the `ase.Atoms.rotate` method

        e.g. Rotating 90degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [[90.0,'z'],[45.0,'y']]

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

    m:
        Atoms object of the generated molecule object within a references directory
    """

    if rotations is None:
        rotations = [[0.0, "x"]]

    m = None

    if mol in chemical_symbols:
        m = Atoms(mol)
        m.cell = cell
        m.center()

    elif mol in g2.names and mol not in ["OH", "NH2", "NH"]:
        m = molecule(mol)
        for r in rotations:
            m.rotate(r[0], r[1])
        lowest_mol = np.min(m.positions[:, 2])
        for atom in m:
            atom.position[2] -= lowest_mol
        m.cell = cell
        m.center()

    elif mol in nrr_intermediate_names:
        m = nrr_mols[mol].copy()
        for r in rotations:
            m.rotate(r[0], r[1])
        m.cell = cell
        m.center()

    elif mol in orr_intermediate_names:
        m = orr_mols[mol].copy()
        for r in rotations:
            m.rotate(r[0], r[1])
        m.cell = cell
        m.center()

    if write_to_disk:
        dir_path = os.path.join(write_location, f"references/{mol}")
        os.makedirs(dir_path, exist_ok=dirs_exist_ok)
        traj_file_path = os.path.join(dir_path, "input.traj")
        m.write(traj_file_path)
        print(f"{mol} structure written to {traj_file_path}")

    return m


def find_adsorption_sites(surface: Union[Atoms, str], ads_site_type: List[str] = None):
    """
    Wrapper for `pymatgen.analysis.adsorption.AdsorbateSiteFinder.find_adsorption_sites`
    which takes in an ase object and returns all of the identified sites in a dictionary
    
    Parameters
    ----------

    surface:
        Atoms object or name of file containing structure 
        as a string specifying the surface for which the symmetry surface 
        sites should be identified

    ads_site_type:
        List of adsorption site types to be searched for.
        Options are ontop, bridge, and hollow

    Returns
    -------

    sites:
        Dictionary containing the reference structures with the identified
        sites for each desired site type
    """

    if ads_site_type is None:
        ads_site_type = ["ontop", "bridge", "hollow"]

    # Reads in surface if necessary
    if type(surface) is str:
        surf = read(surface)

    elif type(surface) is Atoms:
        surf = surface.copy()

    else:
        raise TypeError(
            "surface parameter needs to be either a str or ase.Atoms object"
        )

    conv = (
        AseAtomsAdaptor()
    )  # initialize convertor from ase object to pymatgen structure

    struct = conv.get_structure(surf)  # make conversion to mg structure object

    finder = AdsorbateSiteFinder(struct)  # define site finder

    sites = finder.find_adsorption_sites(positions=ads_site_type, symm_reduce=0.05)

    return sites


def get_adsorption_sites(
    surface: Union[Atoms, str],
    ads_site_type: List[str] = None,
    supcell: Union[Tuple[int], List[int]] = (1, 1, 1),
    view_im: bool = False,
    write_to_disk: bool = False,
    write_to_disk_format: str = "traj",
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    From given surface, gets each of the sites for the ads_site_type specified.
    Writes reference structures containing all of the identified sites
    if specified. Can also visualize all of the identified sites by
    automatically calling the ase-gui
    
    Parameters
    ----------

    surface: 
        Atoms object or name of file containing structure 
        as a string specifying the surface for which the symmetry surface 
        sites should be identified

    ads_site_type:
        List of adsorption site types to be searched for.
        Options are ontop, bridge, and hollow

    supcell:
        Tuple or List specifying the dimension if a supercell
        should be made of the input surface.
        Defaults to searching for sites on the input surface without
        any repetition.

    view_im: 
        Boolean specifying if the sites are to be viewed using the `ase-gui`.
        If True, will automatically call `ase.visualize.view` to visualize
        the identified sites

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_to_disk_format:
        String specifying the format that the references structures should
        be written out to which will be fed into the `ase write` function.
        Defaults to a traj format

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

    sites:
        Dictionary containing the reference structures with the identified
        sites for each desired site type
    """
    # Gets the adsorption sites
    sites = find_adsorption_sites(surface, ads_site_type)

    if ads_site_type is None:
        ads_site_type = ["ontop", "bridge", "hollow"]

    # Reads in surface if necessary
    if type(surface) is str:
        ase_obj = read(surface)

    elif type(surface) is Atoms:
        ase_obj = surface.copy()

    else:
        raise TypeError(
            "surface parameter needs to be either a str or ase.Atoms object"
        )

    name = ase_obj.get_chemical_formula()

    ase_obj.set_constraint()  # Ensures that any constraints are removed for visualization purposes

    for ads_type in ads_site_type:  # Iterates over site type
        ase_obj_i = ase_obj.copy()
        for site in sites[ads_type]:  # Iterates over site given type
            ase_obj_i.append(Atom("X", position=site))  # Adds placeholder atom at site

        ase_obj_with_sites = ase_obj_i * supcell

        if view_im:  # Visualizes each site type
            view(ase_obj_with_sites)

        traj_file_path = None
        if write_to_disk:
            dir_path = os.path.join(write_location, "identified_sites/" + ads_type)
            os.makedirs(dir_path, exist_ok=dirs_exist_ok)
            traj_file_path = os.path.join(
                dir_path,
                str(name) + "_" + str(ads_type) + "_sites." + write_to_disk_format,
            )
            ase_obj_with_sites.write(traj_file_path)
            print(
                f"Reference structure for {str(ads_type)} written to {traj_file_path}"
            )

    return sites


# def get_ads_struct(sub_file, mol, write_traj=False):
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
