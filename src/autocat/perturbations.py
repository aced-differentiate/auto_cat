from ase.io import read, write
from ase import Atoms
from typing import List
from typing import Union
from typing import Dict
import numpy as np
import os
import json


class AutocatPerturbationError(Exception):
    pass


def generate_perturbed_dataset(
    base_structures: List[Atoms],
    minimum_perturbation_distance: float = 0.1,
    maximum_perturbation_distance: float = 1.0,
    maximum_adsorbate_size: int = None,
    num_of_perturbations: int = 10,
    write_to_disk: bool = False,
    write_location: str = ".",
    dirs_exist_ok: bool = False,
):
    """

    Generates a dataset consisting of perturbed structures from
    a base list of structures and keeps track of displacement
    vectors

    Atoms to be perturbed are specified by their `tag`.
    The options for constraints are also set using
    `tag`s with the options as follows:
     0: free in all directions
    -1: free in z only
    -2: free in xy only
    -3: free in x only
    -4: free in y only

    Parameters
    ----------

    base_structures:
        List of Atoms objects or name of file containing structure
        as a strings specifying the base structures to be
        perturbed

    minimum_perturbation_distance:
        Float of minimum acceptable perturbation distance
        Default: 0.1 Angstrom

    maximum_perturbation_distance:
        Float of maximum acceptable perturbation distance
        Default: 1.0 Angstrom

    maximum_adsorbate_size:
        Integer giving the largest number of atoms in an adsorbate
        that should be able to be considered. Used to obtain shape
        of collected matrices
        Default: number of atoms in largest base structure

    num_of_perturbations:
        Int specifying number of perturbations to generate.
        Default 10

    write_to_disk:
        Boolean specifying whether the perturbed structures generated should be
        written to disk.
        Defaults to False.

    write_location:
        String with the location where the perturbed structure
        files written to disk.

    dirs_exist_ok:
        Boolean specifying whether existing directories/files should be
        overwritten or not. This is passed on to the `os.makedirs` builtin.
        Defaults to False (raises an error if directories corresponding the
        species and crystal structure already exist).


    Returns
    -------

    perturbed_dict:
        Dictionary containing all generated perturbed structures
        with their corresponding perturbation matrices

    """

    perturbed_dict = {}

    corrections_list = []
    collected_structure_paths = []
    collected_structures = []

    # loop over each base structure
    for structure_index, structure in enumerate(base_structures):
        # get name of each base structure
        if isinstance(structure, Atoms):
            name = structure.get_chemical_formula() + "_" + str(structure_index)
        elif isinstance(structure, str):
            name = ".".join(structure.split(".")[:-1])
        else:
            raise TypeError(f"Structure needs to be either a str or ase.Atoms object")

        # make sure no base_structures with the same name
        if name in perturbed_dict:
            msg = f"Multiple input base structures named {name}"
            raise AutocatPerturbationError(msg)

        # apply perturbations
        perturbed_dict[name] = {}
        for i in range(num_of_perturbations):
            perturbed_dict[name][str(i)] = perturb_structure(
                structure,
                minimum_perturbation_distance=minimum_perturbation_distance,
                maximum_perturbation_distance=maximum_perturbation_distance,
            )
            # keeps flattened atomic coordinates difference vector
            corrections_list.append(
                perturbed_dict[name][str(i)]["perturbation_matrix"].flatten()
            )

            traj_file_path = None
            pert_mat_file_path = None
            if write_to_disk:
                dir_path = os.path.join(write_location, f"{name}/{str(i)}")
                os.makedirs(dir_path, exist_ok=dirs_exist_ok)
                traj_file_path = os.path.join(dir_path, f"perturbed_structure.traj")
                # write perturbed structure to disk
                perturbed_dict[name][str(i)]["structure"].write(traj_file_path)
                print(
                    f"{name} perturbed structure {str(i)} written to {traj_file_path}"
                )
                pert_mat_file_path = os.path.join(dir_path, "perturbation_matrix.json")
                with open(pert_mat_file_path, "w") as f:
                    # convert to np.array to list for json
                    pert_mat_list = perturbed_dict[name][str(i)][
                        "perturbation_matrix"
                    ].tolist()
                    # write perturbation matrix to json
                    json.dump(pert_mat_list, f)
                print(
                    f"{name} perturbed matrix {str(i)} written to {pert_mat_file_path}"
                )
            # update output dictionary with write paths
            perturbed_dict[name][str(i)].update({"traj_file_path": traj_file_path})
            perturbed_dict[name][str(i)].update(
                {"pert_mat_file_path": pert_mat_file_path}
            )
            # Collects all of the structures into a single list in the same
            # order as the collected matrix rows
            collected_structures.append(perturbed_dict[name][str(i)]["structure"])
            collected_structure_paths.append(traj_file_path)

    if maximum_adsorbate_size is None:
        # find flattened length of largest structure
        adsorbate_sizes = []
        for struct in base_structures:
            adsorbate_sizes.append(len(np.where(struct.get_tags() <= 0)[0]))
        largest_size = 3 * max(adsorbate_sizes)
    else:
        # factor of 3 from flattening (ie. x,y,z)
        largest_size = 3 * maximum_adsorbate_size
    # ensures correct sized padding
    corrections_matrix = np.zeros((len(corrections_list), largest_size))
    # substitute in collected matrices for each row
    for idx, row in enumerate(corrections_list):
        corrections_matrix[idx, : len(row)] = row

    # adds collected matrices to dict that will be returned
    perturbed_dict["correction_matrix"] = corrections_matrix
    correction_matrix_path = None
    # write matrix to disk as json if desired
    if write_to_disk:
        correction_matrix_path = os.path.join(write_location, "correction_matrix.json")
        coll = perturbed_dict["correction_matrix"].tolist()
        with open(correction_matrix_path, "w") as f:
            json.dump(coll, f)
        print(f"Correction matrix written to {correction_matrix_path}")
    # update output dict with collected structures and paths
    perturbed_dict.update(
        {
            "correction_matrix_path": correction_matrix_path,
            "corrections_list": corrections_list,
            "collected_structures": collected_structures,
            "collected_structure_paths": collected_structure_paths,
        }
    )

    return perturbed_dict


def perturb_structure(
    base_structure: Union[str, Atoms],
    minimum_perturbation_distance: float = 0.1,
    maximum_perturbation_distance: float = 1.0,
):
    """

    Perturbs specific atoms in a given structure and keeps
    track of the displacement vectors of each displaced atom

    Atoms to be perturbed are specified by their `tag`.
    The options for constraints are also set using
    `tag`s with the options as follows:
     0: free in all directions
    -1: free in z only
    -2: free in xy only
    -3: free in x only
    -4: free in y only

    Parameters
    ----------

    base_structure:
        Atoms object or name of file containing structure
        as a string specifying the base structure to be
        perturbed

    minimum_perturbation_distance:
        Float of minimum acceptable perturbation distance
        Default: 0.1 Angstrom

    maximum_perturbation_distance:
        Float of maximum acceptable perturbation distance
        Default: 1.0 Angstrom

    Returns
    -------

    perturb_dictionary:
        Dictionary with perturbed structure and displacement vectors

    """
    if isinstance(base_structure, Atoms):
        ase_obj = base_structure.copy()
    elif isinstance(base_structure, str):
        ase_obj = read(base_structure)
    else:
        raise TypeError("base_structure needs to be either a str or ase.Atoms object")

    pert_matrix = np.zeros(ase_obj.positions.shape)

    atom_indices_to_perturb = np.where(ase_obj.get_tags() <= 0)[0].tolist()

    constr = [
        [True, True, True],  # free
        [False, True, False],  # y only
        [True, False, False],  # x only
        [True, True, False],  # xy only
        [False, False, True],  # z only
    ]

    for idx in atom_indices_to_perturb:
        # randomize +/- direction of each perturbation
        signs = np.array([-1, -1, -1]) ** np.random.randint(low=1, high=11, size=(1, 3))
        # generate perturbation matrix
        pert_matrix[idx, :] = (
            constr[ase_obj[idx].tag]
            * signs
            * np.random.uniform(
                low=minimum_perturbation_distance,
                high=maximum_perturbation_distance,
                size=(1, 3),
            )
        )

    ase_obj.positions += pert_matrix

    return {
        "structure": ase_obj,
        "perturbation_matrix": pert_matrix[atom_indices_to_perturb],
    }
