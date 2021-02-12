import os
import numpy as np
from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import Union

from ase.io import read, write
from ase import Atom, Atoms
from ase.visualize import view
from ase.build import fcc100, fcc110, fcc111
from ase.build import bcc100, bcc110, bcc111
from ase.data import atomic_numbers, ground_state_magnetic_moments, reference_states
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from autocat.surface import generate_surface_structures


# TODO(@hegdevinayi): Typing for the returned data
def generate_saa_structures(
    host_species: List[str],
    dopant_species: List[str],
    crystal_structures: Dict[str, str] = None,
    facets: Dict[str, str] = None,
    supercell_dim: Union[Tuple[int], List[int]] = (3, 3, 4),
    default_lat_param_lib: str = None,
    a_dict: Optional[Dict[str, float]] = None,
    c_dict: Optional[Dict[str, float]] = None,
    set_host_magnetic_moments: List[str] = None,
    host_magnetic_moments: Optional[Dict[str, float]] = None,
    set_dopant_magnetic_moments: List[str] = None,
    dopant_magnetic_moments: Optional[Dict[str, float]] = None,
    vacuum: float = 10.0,
    n_fixed_layers: int = 0,
    place_dopant_at_center: bool = True,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    Builds single-atom alloys for all combinations of host species and dopant
    species given. Will write the structures to separate directories if
    specified.

    Parameters
    ----------

    host_species:
        List of chemical species of desired host species.

    dopant_species:
        List of chemical symbols of desired single-atom dopant species.

    crystal_structures:
        Dictionary with crystal structure to be used for each species.
        Options are fcc, bcc, or hcp. If not specified, will use the
        default reference crystal for each species from `ase.data`.

    facets:
        Dictionary with the surface facets to be considered for each
        species.
        If not specified for a given species, the following
        defaults will be used based on the crystal structure:
        fcc/bcc: 100, 111, 110
        hcp: 0001

    supercell_dim:
        Tuple or List specifying the size of the supercell to be
        generated in the format (nx,ny,nz).

    default_lat_param_lib:
        String indicating which library the lattice constants should be pulled
        from if not specified in either a_dict or c_dict. Defaults to ase.

        Options are:
        pbe_fd: parameters calculated using xc=pbe and finite-difference
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and finite-difference
        pbe_pw: parameters calculated using xc=pbe and a plane-wave basis set
        beefvdw_fd: parameters calculated using xc=BEEF-vdW and a plane-wave basis set

        N.B. if there is a species present in species_list that is NOT in the
        reference library specified, it will be pulled from `ase.data`

    a_dict:
        Dictionary with lattice parameters <a> to be used for each species.
        If not specified, defaults from the default_lat_param_lib are used.

    c_dict:
        Dictionary with lattice parameters <c> to be used for each species.
        If not specified, defaults from the default_lat_param_lib module are used.

    set_host_magnetic_moments:
        List of host species for which magnetic moments need to be set.
        If not specified, magnetic moments will be set only for Fe, Co, Ni
        (the ferromagnetic elements).

    host_magnetic_moments:
        Dictionary with the magnetic moments to be set for the host chemical
        species listed previously.
        If not specified, default ground state magnetic moments from
        `ase.data` are used.

    set_dopant_magnetic_moments:
        List of single-atom species for which magnetic moments need to be set.
        If not specified, magnetic moments will guessed for all dopant_species from
        `ase.data`.

    dopant_magnetic_moments:
        Dictionary with the magnetic moments to be set for the single-atom
        dopant species listed previously.
        If not specified, default ground state magnetic moments from
        `ase.data` are used.

    vacuum:
        Float specifying the amount of vacuum to be added on each
        side of the slab.

    n_fixed_layers:
        Integer giving the number of layers of the slab to be fix
        starting from the bottom up. (e.g. a value of 2 will fix the
        bottom 2 layers)

    place_dopant_at_center:
        Boolean specifying that the single-atom should be placed
        at the center of the unit cell if True. If False will leave
        the single-atom at the origin

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.
        In the specified write_location, the following directory structure
        will be created:
        [host]/[dopant]/[facet]/substrate/input.traj

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    Dictionary containing the generated single-atom alloy structures.
    Organized by host -> sa -> facet.
    """

    hosts = generate_surface_structures(
        host_species,
        crystal_structures=crystal_structures,
        facets=facets,
        supercell_dim=supercell_dim,
        default_lat_param_lib=default_lat_param_lib,
        a_dict=a_dict,
        c_dict=c_dict,
        set_magnetic_moments=set_host_magnetic_moments,
        magnetic_moments=host_magnetic_moments,
        vacuum=vacuum,
        n_fixed_layers=n_fixed_layers,
    )

    if set_dopant_magnetic_moments is None:
        set_dopant_magnetic_moments = dopant_species
    if dopant_magnetic_moments is None:
        dopant_magnetic_moments = {}

    dop_mm_library = {
        dop: ground_state_magnetic_moments[atomic_numbers[dop]]
        for dop in dopant_species
    }
    dop_mm_library.update(dopant_magnetic_moments)

    saa_dict = {}
    # iterate over hosts
    for host in hosts:
        saa_dict[host] = {}
        # iterate over single-atoms
        for dopant in dopant_species:
            # ensure host != single-atom
            if dopant != host:
                saa_dict[host][dopant] = {}
                # iterate over surface facets
                for facet in hosts[host]:
                    host_structure = hosts[host][facet].get("structure")
                    doped_structures = generate_doped_structures(
                        host_structure=host_structure,
                        dopant=dopant,
                        place_dopant_at_center=place_dopant_at_center,
                        dopant_magnetic_moment=dop_mm_library.get(dopant),
                    )
                    # pull the structure
                    doped_structure = list(doped_structures.values())[0]["structure"]

                    traj_file_path = None
                    if write_to_disk:
                        dir_path = os.path.join(
                            write_location, host, dopant, facet, "substrate"
                        )
                        os.makedirs(dir_path, exist_ok=dirs_exist_ok)
                        traj_file_path = os.path.join(dir_path, "input.traj")
                        doped_structure.write(traj_file_path)
                        print(
                            f"{dopant}1/{host}({facet}) structure written to {traj_file_path}"
                        )

                    saa_dict[host][dopant][facet] = {
                        "structure": doped_structure,
                        "traj_file_path": traj_file_path,
                    }
    return saa_dict


def generate_doped_structures(
    sub_ase: Atoms,
    dop: str,
    place_dopant_at_center: bool = True,
    dopant_magnetic_moment: float = 0.0,
    all_possible_configs: bool = True,
    sub_both_sides: bool = False,
    target_species: List[str] = None,
    range_tol: float = 0.01,
    dist_from_surf: float = 0.0,
    target_indices: List[int] = None,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """
    Generates doped structures given host material and a dopant by either specifying
    all atom indices to be substituted, or for all substitions to be enumerated via
    `pymatgen.analysis.adsorption.AdsorbateSiteFinder.generate_substitution_structures`

    If specified will write to separate directories for each generated doped system
    organized by target indices.

    Parameters
    ----------

    sub_ase:
        ase.Atoms object of the host slab to be doped

    dop:
        String of the dopant species to be introduced into the system

    place_dopant_at_center:
        Boolean specifying that the single-atom dopant should be placed
        at the center of the unit cell if True.

    dopant_magnetic_moment:
        Float of initial magnetic moment attributed to the doped single-atom.
        Will default to no spin polarization

    all_possible_configs:
        Boolean where if True will enumerate all possible
        configurations via
        `pymatgen.analysis.adsorption.AdsorbateSiteFinder.generate_substitution_structures`

    sub_both_sides:
        Boolean specifying whether to dope both surfaces of the slab during enumeration.
        Will be passed on to
        `pymatgen.analysis.adsorption.AdsorbateSiteFinder.generate_substitution_structures`

        Only available when all_possible_configs is True.

    target_species:
        List of all species that the dopant should substitute during enumeration.
        Will be passed on to
        `pymatgen.analysis.adsorption.AdsorbateSiteFinder.generate_substitution_structures`

    range_tol:
        Float indicating tolerance applied to dist_from_surf when identifying viable targets
        during enumeration.
        Will be passed on to
        `pymatgen.analysis.adsorption.AdsorbateSiteFinder.generate_substitution_structures`

    dist_from_surf:
        Float indicating allowable distance from surface when searching for viable target
        substitutions.
        Will be passed on to
        `pymatgen.analysis.adsorption.AdsorbateSiteFinder.generate_substitution_structures`

    target_indices:
        List of atom indices in the host slab for which substitutions should be made

    write_to_disk:
        Boolean specifying whether the bulk structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the per-species/per-crystal structure
        directories must be constructed and structure files written to disk.

        In the specified write_location, the following directory structure
        will be created:
        [host]_[dopant]_[atom index substituted]/input.traj

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).

    Returns
    -------

    all_ase_structs:
        Dictionary with doped structures (as `ase.Atoms` objects) and write
        location (if-any) for each generated doped structure.

    """
    name = "".join(np.unique(sub_ase.symbols))
    tags = sub_ase.get_tags()
    constr = sub_ase.constraints
    host_mag = sub_ase.get_initial_magnetic_moments()

    all_ase_structs = []

    if all_possible_configs:
        conv = AseAtomsAdaptor()  # converter between pymatgen and ase

        struct = conv.get_structure(
            sub_ase
        )  # convert ase substrate to pymatgen structure

        finder = AdsorbateSiteFinder(struct)

        # collect all substitution structures
        all_structs = finder.generate_substitution_structures(
            dop,
            sub_both_sides=sub_both_sides,
            target_species=target_species,
            range_tol=range_tol,
            dist_from_surf=dist_from_surf,
        )

        i = 0
        while i < len(all_structs):
            all_ase_structs.append(conv.get_atoms(all_structs[i]))
            i += 1

    else:
        if target_indices is None:
            target_indices = [0]

        for index in target_indices:
            struct = sub_ase.copy()
            struct[index].symbol = dop
            all_ase_structs.append(struct)

    all_ase_dict = {}

    i = 0
    while i < len(all_ase_structs):
        ase_struct = all_ase_structs[i]
        ase_struct.set_tags(tags)
        ase_struct.pbc = (1, 1, 0)  # ensure pbc in xy only
        ase_struct.constraints = constr  # propagate constraints
        ase_struct.set_initial_magnetic_moments(
            host_mag
        )  # propagate host magnetization
        sa_ind = _find_sa_ind(ase_struct, dop)
        ase_struct[sa_ind].magmom = dopant_magnetic_moment  # set initial magmom
        if place_dopant_at_center:  # centers the sa
            cent_x = ase_struct.cell[0][0] / 2 + ase_struct.cell[1][0] / 2
            cent_y = ase_struct.cell[0][1] / 2 + ase_struct.cell[1][1] / 2
            cent = (cent_x, cent_y, 0)
            ase_struct.translate(cent)
            ase_struct.wrap()

        traj_file_path = None
        if write_to_disk:
            dir_path = os.path.join(
                write_location, name + "_" + dop + "_" + str(sa_ind)
            )
            os.makedirs(dir_path, exist_ok=dirs_exist_ok)
            traj_file_path = os.path.join(dir_path, "input.traj")
            ase_struct.write(traj_file_path)
            print(f"{name}_{dop}_{str(sa_ind)} structure written to {traj_file_path}")
        i += 1

        all_ase_dict[str(sa_ind)] = {
            "structure": ase_struct,
            "traj_file_path": traj_file_path,
        }

    return all_ase_dict


def _find_sa_ind(saa, dop):
    """ Helper function for finding the index of the single atom """
    syms = np.array(saa.symbols)
    #    TODO tweak to raise warning if multiple instances
    #    unique, counts = np.unique(syms, return_counts=True)
    #    ind = np.where(syms == unique[np.argmin(counts)])[0][
    #        0
    #    ]  # Finds index of species with lowest count
    ind = np.where(syms == dop)
    return ind[0][0]
