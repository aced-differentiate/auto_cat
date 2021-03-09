import numpy as np

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms

from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge

from autocat.learning.featurizers import get_X
from autocat.learning.featurizers import catalyst_featurization

Regressor = Union[BayesianRidge, KernelRidge]


def predict_initial_configuration(
    inital_structure_guess: Atoms,
    adsorbate_indices: List[int],
    trained_regressor_model: Regressor,
    featurization_kwargs: Dict[str, float] = None,
):
    """
    From a trained model, will predict corrected structure
    of a given initial structure guess

    Parameters
    ----------

    initial_structure_guess:
        Atoms object of an initial guess for an adsorbate
        on a surface to be optimized

    adsorbate_indices:
        List of ints giving the atomic indices of the adsorbate
        atoms that can be perturbed

    trained_regressor_model:
        Fit sklearn regression model to be used for prediction

    Returns
    -------

    predicted_correction_matrix:
        Matrix of predicted corrections that were applied

    uncertainty_estimate:
        Standard deviation of prediction from regressor
        (only supported for `bayes` at present)

    corrected_structure:
        Atoms object with corrections applied

    """
    featurized_input = catalyst_featurization(
        inital_structure_guess, **featurization_kwargs
    )

    if isinstance(trained_regressor_model, BayesianRidge):
        (
            flat_predicted_correction_matrix,
            uncertainty_estimate,
        ) = trained_regressor_model.predict(featurized_input, return_std=True)
        predicted_correction_matrix = flat_predicted_correction_matrix.reshape(-1, 3)

    elif isinstance(trained_regressor_model, KernelRidge):
        uncertainty_estimate = 0.0
        flat_predicted_correction_matrix = trained_regressor_model.predict(
            featurized_input
        )
        predicted_correction_matrix = flat_predicted_correction_matrix.reshape(-1, 3)

    else:
        raise TypeError("Trained Regressor Model is not a supported type")

    corrected_structure = inital_structure_guess.copy()
    corrected_structure.positions += predicted_correction_matrix

    return predicted_correction_matrix, uncertainty_estimate, corrected_structure


def get_trained_model_on_perturbed_systems(
    perturbed_structures: List[Union[Atoms, str]],
    adsorbate_indices_dictionary: Dict[str, int],
    collected_matrices: np.ndarray,
    model_name: str = "bayes",
    structure_featurizer: str = "sine_matrix",
    adsorbate_featurizer: str = "soap",
    featurization_kwargs: Dict = None,
    model_kwargs: Dict = None,
):
    """
    Given a list of base_structures, will generate perturbed structures
    and train a regression model on them

    Parameters
    ----------

    perturbed_structures:
        List of perturbed structures to be trained upon

    adsorbate_indices_dictionary:
        Dictionary mapping structures to desired adsorbate_indices
        (N.B. if structure is given as an ase.Atoms object,
        the key for this dictionary should be
        f"{structure.get_chemical_formula()}_{index_in_`perturbed_structures`}")

    collected_matrices:
        Numpy array of collected matrices of perturbations corresponding to
        each of the perturbed structures.
        This can be generated via `autocat.perturbations.generate_perturbed_dataset`.
        Shape should be (# of structures, 3 * # of atoms in the largest structure)

    model_name:
        String giving the name of the `sklearn` regression model to use.
        Options:
        - bayes: Bayesian Ridge Regression
        - krr: Kernel Ridge Regression

    structure_featurizer:
        String giving featurizer to be used for full structure which will be
        fed into `autocat.learning.featurizers.full_structure_featurization`

    adsorbate_featurizer:
        String giving featurizer to be used for full structure which will be
        fed into `autocat.learning.featurizers.adsorbate_structure_featurization`

    Returns
    -------

    trained_model:
        Trained `sklearn` model object
    """
    X = get_X(
        perturbed_structures,
        adsorbate_indices_dictionary=adsorbate_indices_dictionary,
        structure_featurizer=structure_featurizer,
        adsorbate_featurizer=adsorbate_featurizer,
        **featurization_kwargs
    )

    regressor = _get_regressor(model_name, **model_kwargs)

    regressor.fit(X, collected_matrices)

    return regressor


def _get_regressor(model_name: str, **kwargs):
    """
    Gets `sklearn` regressor object
    """
    if model_name == "bayes":
        return BayesianRidge(**kwargs)
    elif model_name == "krr":
        return KernelRidge(**kwargs)
    else:
        raise NotImplementedError("model selected not implemented")
