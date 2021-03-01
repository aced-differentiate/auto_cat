import os
import numpy as np
from typing import List
from typing import Tuple
from typing import Dict
from typing import Union
from collections.abc import Sequence

from ase import Atom, Atoms
from ase.io import read
from ase.build import add_adsorbate
from ase.build import molecule as build_molecule
from ase.visualize import view
from ase.data import chemical_symbols
from ase.data import atomic_numbers
from ase.data import covalent_radii
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index, VoronoiNN

from autocat.data.intermediates import NRR_MOLS
from autocat.data.intermediates import NRR_INTERMEDIATE_NAMES
from autocat.data.intermediates import ORR_MOLS
from autocat.data.intermediates import ORR_INTERMEDIATE_NAMES


def generate_rxn_structures(
    surf: Union[str, Atoms],
    sites: Dict[str, List[Union[Tuple[float], List[float]]]] = None,
    all_sym_sites: bool = True,
    site_type: List[str] = None,
    ads: List[Union[str, Atoms]] = None,
    height: Dict[str, float] = None,
    mol_indices: Dict[str, int] = None,
    rots: Dict[str, List[List[Union[float, str]]]] = None,
    site_im: bool = False,
    refs: List[str] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """

    Builds structures for reaction intermediates given a surface and list of adatoms

    Parameters
    ----------

    surf:
        Atoms object or name of file containing structure 
        as a string specifying the surface for which the adsorbate should be placed

    sites:
        Dictionary of list of sites to be considered with the keys being user defined labels
        (e.g. {'custom': [(0.0,0.0),(1.5,1.5)]})

    all_sym_sites:
        Bool specifying if all sites identified by
        `pymatgen.analysis.adsorption.AdsorbateSiteFinder.find_adsorption_sites`.

        If True, overrides any sites defined in `sites` and uses all identified sites

    site_type:
        List of adsorption site types to be searched for.
        Options are ontop, bridge, and hollow

    ads:
        List of names of adsorbates or Atoms objects to be placed on the surface.
        Defaults to placing only H

    height:
        Float specifying the height above surface where adsorbate should be initially placed.
        
        If adsorbate given as an Atoms object, the key here should be specified using the result
        of `adsorbate.get_chemical_formula()`.
        e.g. ads=[adsorbate], height={adsorbate.get_chemical_formala():2.0}

    mol_indices:
        Dictionary specifying the molecule index to be used as the reference point
        for the placement of each adsorbate. Will be fed into `ase.build.add_adsorbate` function.
        Defaults to the molecule at index 0 for each adsorbate

        If adsorbate given as an Atoms object, the key here should be specified using the result
        of `adsorbate.get_chemical_formula()`.
        e.g. ads=[adsorbate], mol_indicies={adsorbate.get_chemical_formala():1}

    rots:
        Dictionary of rotations to be applied to each adatom in `ads`.
        Defaults to no rotations applied

        If adsorbate given as an Atoms object, the key here should be specified using the result
        of `adsorbate.get_chemical_formula()`.
        e.g. ads=[adsorbate], rots={adsorbate.get_chemical_formala():[[90.0,'y']]}

    site_im:
        Boolean specifying if reference structures showing all of the automatically
        identified sites should be written to disk
            
    refs:
        List of reference molecular structures to be generated if specified
        e.g. H2, H2O molecules for ORR

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.
        In the specified write_location, the following directory structure
        will be created:
        TODO: update according to taxonomy

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).
            
    Returns
    -------

    rxn_structs:
        Dictionary containing all of the reaction structures (and reference
        states if specified)

    """

    if ads is None:
        ads = ["H"]

    if height is None:
        height = {}

    if mol_indices is None:
        mol_indices = {}

    if rots is None:
        rots = {}

    ads_dict = {}
    for a in ads:
        if type(a) is str:
            ads_dict[a] = a
        else:
            ads_dict[a.get_chemical_formula()] = a

    height_library = {a: None for a in ads_dict}
    height_library.update(height)

    mol_indices_library = {a: 0 for a in ads_dict}
    mol_indices_library.update(mol_indices)

    rots_library = {a: [[0.0, "x"]] for a in ads_dict}
    rots_library.update(rots)

    if all_sym_sites:
        # gets identified sites as dict
        sites = get_adsorption_sites(
            surf, ads_site_type=site_type, write_to_disk=site_im
        )

    if sites is None:
        sites = {"origin": [(0.0, 0.0)]}

    rxn_structs = {}
    for a in ads_dict:
        rxn_structs[a] = {}
        for typ in sites:
            if typ != "all":
                rxn_structs[a][typ] = {}
                for p in sites[typ]:
                    rpos = np.around(p, 3)
                    loc = str(rpos[0]) + "_" + str(rpos[1])
                    st = place_adsorbate(
                        surf,
                        mol=ads_dict[a],
                        write_to_disk=write_to_disk,
                        write_location=write_location,
                        dirs_exist_ok=dirs_exist_ok,
                        rotations=rots_library[a],
                        mol_index=mol_indices_library[a],
                        height=height_library[a],
                        position=p[:2],
                        label=typ,
                    )
                    rxn_structs[a][typ][loc] = {
                        "structure": st[typ].get("structure"),
                        "traj_file_path": st[typ].get("traj_file_path"),
                    }

    if refs is None:
        # Defaults to no references added, setting to {} will skip the later for loop
        refs = {}

    else:
        # Since references specified, initializes appropriate key for dict to be returned
        rxn_structs["references"] = {}

    for ref in refs:
        r = generate_molecule(
            ref,
            write_to_disk=write_to_disk,
            write_location=write_location,
            dirs_exist_ok=dirs_exist_ok,
        )
        rxn_structs["references"][ref] = r

    return rxn_structs


def place_adsorbate(
    surface: Union[str, Atoms],
    mol: Union[str, Atoms],
    position: Union[Tuple[float], List[float]] = (0.0, 0.0),
    height: float = None,
    rotations: List[List[Union[float, str]]] = None,
    mol_index: int = 0,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
    label: str = "custom",
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
        Float specifying the height above surface where adsorbate should be initially placed.
        If None, will guess initial height based on covalent radii of nearest neighbors

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
        adsorbates/[adsorbate1]/[label]/[xy_site_coord1]/input.traj
        adsorbates/[adsorbate1]/[label]/[xy_site_coord2]/input.traj
        ...
        adsorbates/[adsorbate2]/[label]/[xy_site_coord1]/input.traj
        ...

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    label:
        String giving a user specified label of the position when writing
        to disk which will be used in the directory name.

    Returns
    -------

    surf:
        Atoms object of the surface with the molecule adsorbed

    """

    # Identify if surface specified as filename or atoms object
    if isinstance(surface, str):
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
        adsorbate = generate_molecule(mol, rotations=rotations).get("structure")
        name = mol
    elif type(mol) is Atoms:
        adsorbate = mol.copy()
        name = mol.get_chemical_formula()
        for r in rotations:
            adsorbate.rotate(r[0], r[1])

    # If height not given takes educated guess
    if height is None:
        height = get_adsorbate_height_estimate(
            surf, position, adsorbate, mol_index=mol_index
        )

    add_adsorbate(surf, adsorbate, height, position=position, mol_index=mol_index)

    traj_file_path = None
    if write_to_disk:
        rpos = np.around(position, 3)
        dir_path = os.path.join(
            write_location,
            "adsorbates/"
            + name
            + "/"
            + label
            + "/"
            + str(rpos[0])
            + "_"
            + str(rpos[1]),
        )
        os.makedirs(dir_path, exist_ok=dirs_exist_ok)
        traj_file_path = os.path.join(dir_path, "input.traj")
        surf.write(traj_file_path)
        print(f"{name} at ({rpos[0]},{rpos[1]}) written to {traj_file_path}")

    ads_structs = {label: {"structure": surf, "traj_file_path": traj_file_path}}

    return ads_structs


def get_adsorbate_height_estimate(
    surface: Atoms,
    position: Union[Tuple[float], List[float]],
    adsorbate: Atoms,
    mol_index: int = 0,
    scale: float = 1.0,
):
    """
    Guesses initial height for adsorbate placement based on summing covalent radii
    to get the approximate bond length with each nearest neighbor. Takes the average
    height based on each nearest neighbor contribution.

    Parameters
    ----------

    surface:
        Atoms object the surface for which nearest neighbors 
        of the adsorbate should be identified

    position:
        Tuple or list of the xy cartesian coordinates for where the adsorbate would be placed 

    adsorbate:
        Atoms object of adsorbate to be placed on `surface`

    mol_index:
        Integer index of atom in molecule that will be the reference for placing the molecule
        at `position`. Defaults to the atom at index 0

    scale:
        Float giving a scale factor to be applied to the calculated bond length
        e.g. scale=1.1 -> bond length = 1.1*(covalent_radius1 + covalent_radius2)

    Returns
    -------

    height_estimate:
        Float giving estimated good initial height for adsorbate
    """
    nn_list = get_adsorbate_slab_nn_list(surface, position)
    rad_ads = covalent_radii[atomic_numbers[adsorbate[mol_index].symbol]]
    guessed_heights = []
    for nn in nn_list:
        rad_nn = covalent_radii[atomic_numbers[nn[0]]]
        r_dist = scale * (rad_ads + rad_nn)
        position_array = np.array(position)
        if (r_dist ** 2 - np.sum((position_array - nn[1][:2]) ** 2)) >= 0.0:
            height = np.sqrt(r_dist ** 2 - np.sum((position_array - nn[1][:2]) ** 2))
        else:
            height = np.nan
        guessed_heights.append(height)

    if np.isnan(np.nanmean(guessed_heights)):
        height_estimate = 0.0
    else:
        height_estimate = np.nanmean(guessed_heights)

    return height_estimate


def get_adsorbate_slab_nn_list(
    surface: Atoms, position: Union[List[float], Tuple[float]], height: float = 1.5
):
    """
    Gets list of nearest neighbors for the adsorbate on the surface at a given position.

    Parameters
    ----------

    surface:
        Atoms object the surface for which nearest neighbors 
        of the adsorbate should be identified

    position:
        Tuple or list of the xy cartesian coordinates for where the adsorbate would be placed

    Returns
    -------

    nn_list:
        List containing the species and coordinates of each identified n.n. in the form:
        [[species1,[x1,y1]],[species2,[x2,y2]],...]
    """
    surf = surface.copy()
    conv = AseAtomsAdaptor()
    add_adsorbate(surf, "X", height=height, position=position)
    init_guess = conv.get_structure(surf)
    nn = get_neighbors_of_site_with_index(init_guess, -1)
    nn_list = [[nn[i].species.hill_formula, nn[i].coords] for i in range(len(nn))]
    return nn_list


def generate_molecule(
    molecule_name: str,
    rotations: Sequence[Sequence[float, str]] = None,
    cell: Sequence[int] = (15, 15, 15),
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    Generates an `ase.Atoms` object of an isolated molecule.
    If specified, can write out a .traj file containing the isolated molecule
    in a box.

    Parameters
    ----------

    molecule_name:
        String of the name of the molecule to be generated. Will search in
        the `ase` g2 database first, then in `autocat.intermediates`.

    rotations:
        List of rotation operations to be carried out which will be fed into
        the `ase.Atoms.rotate` method.

        e.g. Rotating 90 degrees around the z axis followed by 45 degrees
        around the y-axis can be specified as
            [(90.0, 'z'), (45.0, 'y')]

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

    Dictionary containing Atoms object of the generated molecule and path to
    .traj file if written to disk.
    """

    if rotations is None:
        rotations = [[0.0, "x"]]

    m = None

    # atom in a box
    if molecule_name in chemical_symbols:
        m = Atoms(molecule_name)
        m.cell = cell
        m.center()

    elif molecule_name in g2.names and molecule_name not in ["OH", "NH2", "NH"]:
        m = build_molecule(molecule_name)
        for r in rotations:
            m.rotate(r[0], r[1])
        lowest_z = np.min(m.positions[:, 2])
        for atom in m:
            atom.position[2] -= lowest_z
        m.cell = cell
        m.center()

    elif molecule_name in NRR_INTERMEDIATE_NAMES:
        m = NRR_MOLS[molecule_name].copy()
        for r in rotations:
            m.rotate(r[0], r[1])
        m.cell = cell
        m.center()

    elif molecule_name in ORR_INTERMEDIATE_NAMES:
        m = ORR_MOLS[molecule_name].copy()
        for r in rotations:
            m.rotate(r[0], r[1])
        m.cell = cell
        m.center()

    traj_file_path = None
    if write_to_disk:
        dir_path = os.path.join(write_location, f"references/{molecule_name}")
        os.makedirs(dir_path, exist_ok=dirs_exist_ok)
        traj_file_path = os.path.join(dir_path, "input.traj")
        m.write(traj_file_path)
        print(f"{molecule_name} molecule structure written to {traj_file_path}")

    return {"structure": m, "traj_file_path": traj_file_path}


def find_adsorption_sites(
    surface: Union[Atoms, str], ads_site_type: List[str] = None, **kwargs
):
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

    # TODO: pass kwargs
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

    # constraints mess with ase-gui (test)
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
