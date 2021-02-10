from ase.io import read, write
from ase import Atoms
from typing import List
from typing import Union
from typing import Dict
import numpy as np


def generate_perturbed_dataset(
    base_structures: List[Atoms],
    atom_indices_to_perturb_dictionary: Dict[str, List[int]],
):
    """

    Generates a dataset consisting of perturbed structures from
    a base list of structures and keeps track of displacement
    vectors

    Parameters
    ----------

    base_structures:
        List of Atoms objects or name of file containing structure
        as a strings specifying the base structures to be
        perturbed

    atom_indices_to_perturb_dictionary:
        Dictionary List of atomic indices for the atoms that should
        be perturbed. Keys are each of the provided base structures

    Returns
    -------

    """
    return None


def perturb_structure(
    base_structure: Union[str, Atoms],
    atom_indices_to_perturb: List[int],
    minimum_perturbation_distance: float = 0.1,
    maximum_perturbation_distance: float = 1.0,
):
    """

    Perturbs specific atoms in a given structure and keeps
    track of the displacement vectors of each displaced atom

    Parameters
    ----------

    base_structure:
        Atoms object or name of file containing structure
        as a string specifying the base structure to be
        perturbed

    atom_indices_to_perturb:
        List of atomic indices for the atoms that should
        be perturbed

    minimum_perturbation_distance:
        Float of minimum acceptable perturbation distance

    maximum_perturbation_distance:
        Float of maximum acceptable perturbation distance

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

    for idx in atom_indices_to_perturb:
        # randomize +/- direction of each perturbation
        signs = np.array([-1, -1, -1]) ** np.random.randint(low=1, high=2, size=(1, 3))
        # generate perturbation matrix
        pert_matrix[idx, :] = signs * np.random.uniform(
            low=minimum_perturbation_distance,
            high=maximum_perturbation_distance,
            size=(1, 3),
        )

    ase_obj.positions += pert_matrix

    return {"structure": ase_obj, "perturbation_matrix": pert_matrix}
