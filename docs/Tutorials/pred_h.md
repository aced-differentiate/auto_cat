In this tutorial we are going to show how to use the learning tools within 
AutoCat to train a regressor that can predict adsorption energies of hydrogen 
on a set of single-atom alloys.

## Creating a `DesignSpace`

Let's start by creating a `DesignSpace`. Normally each of these 
structures would be optimized via DFT, but for demo purposes 
we'll use the generated structures directly. First we need to generate the single-atom 
alloys. Here, we can use AutoCat's 
[`generate_saa_structures`](../API/Structure_Generation/saa.md#autocat.saa.generate_saa_structures) 
function. 

```py
>>> # Generate the clean single-atom alloy structures
>>> from autocat.saa import generate_saa_structures
>>> from autocat.utils import extract_structures
>>> saa_struct_dict = generate_saa_structures(
...     ["Fe", "Cu", "Au"],
...     ["Pt", "Pd", "Ni"],
...     facets={"Fe":["110"], "Cu":["111"], "Au":["111"]},
...     n_fixed_layers=2,
... )
>>> saa_structs = extract_structures(saa_struct_dict)
```

Now that we have the clean structures, let's adsorb hydrogen on the surface. 
For convenience let's place H at the origin instead of considering all symmetry sites. 
To accomplish this we can make use of AutoCat's 
[`place_adsorbate`](../API/Structure_Generation/adsorption.md#autocat.adsorption.place_adsorbate)
function.

```py
>>> # Adsorb hydrogen onto each of the generated SAA surfaces
>>> from autocat.adsorption import place_adsorbate
>>> ads_structs = []
>>> for clean_struct in saa_structs:
...     ads_dict = place_adsorbate(
...        clean_struct,
...        "H",
...        (0.,0.)
...     )
...     ads_struct = extract_structures(ads_dict)[0]
...     ads_structs.append(ads_struct)
```

This has collected all of the single-atom alloys with hydrogen adsorbed into 
a single list of `ase.Atoms` objects, `ads_structs`. Ideally at this stage we'd have 
adsorption energies for each of the generated structures after relaxation. As a proxy 
in this demo we'll create random labels, but this should be adsorption energies if you 
want to train a meaningful Predictor!

```py
>>> # Generate the labels for each structure
>>> import numpy as np
>>> labels = np.random.uniform(-1.5,1.5,size=len(ads_structs))
```

Finally, using both our structures and labels we can define a `DesignSpace`. In practice, 
if any of the labels for a structure are unknown, it can be included as a `numpy.nan` 

```py
>>> from autocat.learning.sequential import DesignSpace
>>> design_space = DesignSpace(ads_structs, labels)
```

## Setting up a `Predictor`

When setting up our `Predictor` we now have two choices to make:

1. The technique to be used for featurizing the systems
2. The regression model to be used for training and predictions

Internally, the `Predictor` will contain a `Featurizer` object which contains all of 
our choices for how to featurize the systems. Our choice of featurizer class and 
the associated kwargs are specified via the `featurizer_class` and 
`featurization_kwargs` arguments, respectively. By providing the design space structures 
some of the kwargs related to the featurization (e.g. maximum structure size) can be 
automatically obtained.

Similarly, we can specify the regressor to be used within the `model_class` and 
`model_kwargs` arguments. The class should be "`sklearn`-like" with `fit` and 
`predict` methods.

Let's featurize the hydrogen environment via `dscribe`'s `SOAP` class with 
`sklearn`'s `GaussianProcessRegressor` for regression.

```py
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> from dscribe import SOAP
>>> from autocat.learning.predictors import Predictor
>>> kernel = RBF(1.5)
>>> model_kwargs={"kernel": kernel}
>>> featurization_kwargs={
...     "design_space_structures": design_space.design_space_structures,
...     "kwargs": {"rcut": 7.0, "nmax": 8, "lmax": 8}
... }
>>> predictor = Predictor(
...     model_class=GaussianProcessRegressor,
...     model_kwargs=model_kwargs,
...     featurizer_class=SOAP,
...     featurization_kwargs=featurization_kwargs,
... )
```

## Training and making predictions

With our newly defined `Predictor` we can train it using data from our 
`DesignSpace` and the `fit` method.

```py
>>> train_structures = design_space.design_space_structures[:5]
>>> train_labels = design_space.design_space_labels[:5]
>>> predictor.fit(train_structures, train_labels)
```

Making predictions is a similar process except using the `predict` method.

```py
>>> test_structures = design_space.design_space_structures[5:]
>>> predicted_labels = predictor.predict(test_structures)
```

In this example, since we already have the labels for the test structures, we can 
also use the `score` method to calculate a prediction score.

```py
>>> test_labels = design_space.design_space_labels[5:]
>>> mae = predictor.score(test_structures, test_labels)
```