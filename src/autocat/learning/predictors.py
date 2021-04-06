import numpy as np

from typing import List
from typing import Dict
from typing import Union

from ase import Atoms

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor

from autocat.learning.featurizers import get_X
from autocat.learning.featurizers import _get_number_of_features


class AutoCatStructureCorrector:
    def __init__(
        self,
        model_class=None,
        structure_featurizer: str = None,
        adsorbate_featurizer: str = None,
        maximum_structure_size: int = None,
        maximum_adsorbate_size: int = None,
        species_list: List[str] = None,
        structure_featurization_kwargs: Dict = None,
        adsorbate_featurization_kwargs: Dict = None,
        model_kwargs: Dict = None,
    ):
        """
        Constructor.

        Parameters
        ----------

        model_class:
            Class of regression model to be used for training and prediction.
            If this is changed after initialization, all previously set
            model_kwargs will be removed.
            N.B. must have fit and predict methods

        structure_featurizer:
            String giving featurizer to be used for full structure which will be
            fed into `autocat.learning.featurizers.full_structure_featurization`

        adsorbate_featurizer:
            String giving featurizer to be used for full structure which will be
            fed into `autocat.learning.featurizers.adsorbate_structure_featurization

        maximum_structure_size:
            Size of the largest structure to be supported by the representation.
            Default: number of atoms in largest structure within `structures`

        maximum_adsorbate_size:
            Integer giving the maximum adsorbate size to be encountered
            (ie. this determines if zero-padding should be applied and how much).
            If the provided value is less than the adsorbate size given by
            `adsorbate_indices`, representation will remain size of the adsorbate.
            Default: size of adsorbate provided

        species_list:
            List of species that could be encountered for featurization.
            Default: Parses over all `structures` and collects all encountered species

        """
        self.is_fit = False

        self._model_class = GaussianProcessRegressor
        self.model_class = model_class

        self._model_kwargs = None
        self.model_kwargs = model_kwargs

        self.regressor = self.model_class(self.model_kwargs)

        self._structure_featurizer = None
        self.structure_featurizer = structure_featurizer

        self._adsorbate_featurizer = None
        self.adsorbate_featurizer = adsorbate_featurizer

        self._structure_featurization_kwargs = None
        self.structure_featurization_kwargs = structure_featurization_kwargs

        self._adsorbate_featurization_kwargs = {"rcut": 3.0, "nmax": 4, "lmax": 4}
        self.adsorbate_featurization_kwargs = adsorbate_featurization_kwargs

        self._maximum_structure_size = None
        self.maximum_structure_size = maximum_structure_size

        self._maximum_adsorbate_size = None
        self.maximum_adsorbate_size = maximum_adsorbate_size

        self._species_list = None
        self.species_list = species_list

    @property
    def model_class(self):
        return self._model_class

    @model_class.setter
    def model_class(self, model_class):
        if model_class is not None:
            self._model_class = model_class
            # removes any model kwargs from previous model
            self._model_kwargs = None
            if self.is_fit:
                self.is_fit = False

    @property
    def model_kwargs(self):
        return self._model_kwargs

    @model_kwargs.setter
    def model_kwargs(self, model_kwargs):
        if model_kwargs is not None:
            assert isinstance(model_kwargs, dict)
            if self._model_kwargs is not None:
                self._model_kwargs = model_kwargs
            if self.is_fit:
                self.is_fit = False

    @property
    def structure_featurizer(self):
        return self._structure_featurizer

    @structure_featurizer.setter
    def structure_featurizer(self, structure_featurizer):
        if structure_featurizer is not None:
            self._structure_featurizer = structure_featurizer
            if self.is_fit:
                self.is_fit = False

    @property
    def adsorbate_featurizer(self):
        return self._adsorbate_featurizer

    @adsorbate_featurizer.setter
    def adsorbate_featurizer(self, adsorbate_featurizer):
        if adsorbate_featurizer is not None:
            self._adsorbate_featurizer = adsorbate_featurizer
            if self.is_fit:
                self.is_fit = False

    @property
    def structure_featurization_kwargs(self):
        return self._structure_featurization_kwargs

    @structure_featurization_kwargs.setter
    def structure_featurization_kwargs(self, structure_featurization_kwargs):
        if structure_featurization_kwargs is not None:
            assert isinstance(structure_featurization_kwargs, dict)
            if self._structure_featurization_kwargs is not None:
                self._structure_featurization_kwargs = structure_featurization_kwargs
            if self.is_fit:
                self.is_fit = False

    @property
    def adsorbate_featurization_kwargs(self):
        return self._adsorbate_featurization_kwargs

    @adsorbate_featurization_kwargs.setter
    def adsorbate_featurization_kwargs(self, adsorbate_featurization_kwargs):
        if adsorbate_featurization_kwargs is not None:
            assert isinstance(adsorbate_featurization_kwargs, dict)
            self._adsorbate_featurization_kwargs = adsorbate_featurization_kwargs
            if self.is_fit:
                self.is_fit = False

    @property
    def maximum_structure_size(self):
        return self._maximum_structure_size

    @maximum_structure_size.setter
    def maximum_structure_size(self, maximum_structure_size):
        if maximum_structure_size is not None:
            self._maximum_structure_size = maximum_structure_size
            if self.is_fit:
                self.is_fit = False

    @property
    def maximum_adsorbate_size(self):
        return self._maximum_adsorbate_size

    @maximum_adsorbate_size.setter
    def maximum_adsorbate_size(self, maximum_adsorbate_size):
        if maximum_adsorbate_size is not None:
            self._maximum_adsorbate_size = maximum_adsorbate_size
            if self.is_fit:
                self.is_fit = False

    @property
    def species_list(self):
        return self._species_list

    @species_list.setter
    def species_list(self, species_list):
        if species_list is not None:
            self._species_list = species_list
            if self.is_fit:
                self.is_fit = False

    def get_total_number_of_features(self):
        # get specified kwargs for featurizers
        str_kwargs = self.structure_featurization_kwargs
        ads_kwargs = self.adsorbate_featurization_kwargs
        if str_kwargs is None:
            str_kwargs = {}
        if ads_kwargs is None:
            ads_kwargs = {}
        if self.structure_featurizer is not None:
            # check if one of dscribe structure featurizers
            if self.structure_featurizer in ["sine_matrix", "coulomb_matrix"]:
                str_kwargs.update({"n_atoms_max": self.maximum_structure_size})
            num_struct_feat = _get_number_of_features(
                self.structure_featurizer, **str_kwargs
            )
        else:
            # no structure featurizer present
            num_struct_feat = 0
        if self.adsorbate_featurizer is not None:
            # check if one of dscribe structure featurizers
            if self.adsorbate_featurizer == "soap":
                ads_kwargs.update({"species": self.species_list})
            num_ads_feat = _get_number_of_features(
                self.adsorbate_featurizer, **ads_kwargs
            )
        else:
            # no adsorbate featurizer present
            num_ads_feat = 0
        return num_struct_feat, num_ads_feat

    def fit(
        self,
        perturbed_structures: List[Union[Atoms, str]],
        collected_matrices: np.ndarray,
        adsorbate_indices_dictionary: Dict[str, int] = None,
    ):
        """
        Given a list of perturbed structures
        will featurize and train a regression model on them

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

        Returns
        -------

        trained_model:
            Trained `sklearn` model object
        """
        X = get_X(
            perturbed_structures,
            adsorbate_indices_dictionary=adsorbate_indices_dictionary,
            maximum_structure_size=self.maximum_structure_size,
            structure_featurizer=self.structure_featurizer,
            maximum_adsorbate_size=self.maximum_adsorbate_size,
            adsorbate_featurizer=self.adsorbate_featurizer,
            species_list=self.species_list,
            structure_featurization_kwargs=self.structure_featurization_kwargs,
            adsorbate_featurization_kwargs=self.adsorbate_featurization_kwargs,
        )

        self.regressor.fit(X, collected_matrices)

        if self.maximum_structure_size is None:
            self.maximum_structure_size = max([len(s) for s in perturbed_structures])

        if self.maximum_adsorbate_size is None:
            self.maximum_adsorbate_size = max(
                [
                    len(adsorbate_indices_dictionary[a])
                    for a in adsorbate_indices_dictionary
                ]
            )

        if self.species_list is None:
            species_list = []
            for s in perturbed_structures:
                found_species = np.unique(s.get_chemical_symbols()).tolist()
                new_species = [
                    spec for spec in found_species if spec not in species_list
                ]
                species_list.extend(new_species)
            self.species_list = species_list
        self.is_fit = True

    def predict(
        self,
        initial_structure_guesses: List[Atoms],
        adsorbate_indices_dictionary: Dict[str, int] = None,
    ):
        """
        From a trained model, will predict corrected structure
        of a given initial structure guess

        Parameters
        ----------

        initial_structure_guesses:
            List of Atoms objects of initial guesses for adsorbate
            placement to be optimized

        adsorbate_indices_dictionary:
            Dictionary mapping structures to desired adsorbate_indices
            (N.B. if structures is given as ase.Atoms objects,
            the key for this dictionary should be
            ase.Atoms.get_chemical_formula()+ "_" + str(index in list)

        Returns
        -------

        predicted_correction_matrix:
            Matrix of predicted corrections that were applied

        corrected_structure:
            Atoms object with corrections applied

        """
        assert self.is_fit
        featurized_input = get_X(
            structures=initial_structure_guesses,
            adsorbate_indices_dictionary=adsorbate_indices_dictionary,
            structure_featurizer=self.structure_featurizer,
            adsorbate_featurizer=self.adsorbate_featurizer,
            maximum_structure_size=self.maximum_structure_size,
            maximum_adsorbate_size=self.maximum_adsorbate_size,
            species_list=self.species_list,
            structure_featurization_kwargs=self.structure_featurization_kwargs,
            adsorbate_featurization_kwargs=self.adsorbate_featurization_kwargs,
        )

        predicted_correction_matrix, unc = self.regressor.predict(
            featurized_input, return_std=True
        )

        corrected_structures = [
            init_struct.copy() for init_struct in initial_structure_guesses
        ]

        corrected_structures = []
        for idx, struct in enumerate(initial_structure_guesses):
            cs = struct.copy()
            name = cs.get_chemical_formula() + "_" + str(idx)
            list_of_adsorbate_indices = adsorbate_indices_dictionary[name]
            list_of_adsorbate_indices.sort()
            num_of_adsorbates = len(list_of_adsorbate_indices)
            corr = predicted_correction_matrix[idx, : 3 * num_of_adsorbates].reshape(
                num_of_adsorbates, 3
            )
            cs.positions[list_of_adsorbate_indices] += corr
            corrected_structures.append(cs)

        return predicted_correction_matrix, corrected_structures, unc
