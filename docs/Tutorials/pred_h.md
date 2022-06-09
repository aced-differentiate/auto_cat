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
>>> from autocat.utils import flatten_structures_dict
>>> saa_struct_dict = generate_saa_structures(
...     ["Fe", "Cu", "Au"],
...     ["Pt", "Pd", "Ni"],
...     facets={"Fe":["110"], "Cu":["111"], "Au":["111"]},
...     n_fixed_layers=2,
... )
>>> saa_structs = flatten_structures_dict(saa_struct_dict)
```

Now that we have the clean structures, let's adsorb hydrogen on the surface. 
For convenience let's place H at the origin instead of considering all symmetry sites. 
To accomplish this we can make use of AutoCat's 
[`place_adsorbate`](../API/Structure_Generation/adsorption.md#autocat.adsorption.place_adsorbate)
function.

```py
>>> # Adsorb hydrogen onto each of the generated SAA surfaces
>>> from autocat.adsorption import place_adsorbate
>>> from ase import Atoms
>>> ads_structs = []
>>> for clean_struct in saa_structs:
...     ads_struct = place_adsorbate(clean_struct, Atoms("H"))
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
>>> design_space
+-------------------------+-------------------------------------------+
|                         |                DesignSpace                |
+-------------------------+-------------------------------------------+
|    total # of systems   |                     9                     |
| # of unlabelled systems |                     0                     |
|  unique species present | ['Fe', 'H', 'Pt', 'Pd', 'Ni', 'Cu', 'Au'] |
|      maximum label      |             1.0173326963281424            |
|      minimum label      |            -1.4789390894451206            |
+-------------------------+-------------------------------------------+
```

## Setting up a `Predictor`

When setting up our `Predictor` we now have two choices to make:

1. The technique to be used for featurizing the systems
2. The regression model to be used for training and predictions

Internally, the `Predictor` will contain a `Featurizer` object (that the user supplies) 
which stores all of our choices for how to featurize the systems. Our choice of 
featurizer class and the associated kwargs are specified via the `featurizer_class` and 
`kwargs` arguments, respectively. By providing the design space structures 
some of the kwargs related to the featurization (e.g. maximum structure size) can be 
automatically obtained.

Let's featurize the hydrogen environment via `dscribe`'s `SOAP` class 
```py
>>> from autocat.learning.featurizers import Featurizer
>>> from dscribe.descriptors.soap import SOAP
>>> featurizer = Featurizer(
...     featurizer_class=SOAP,
...     kwargs={"rcut": 7.0, "nmax": 8, "lmax": 8},
...     design_space_structures=design_space.design_space_structures
... )
>>> featurizer
+-----------------------------------+-------------------------------------------+
|                                   |                 Featurizer                |
+-----------------------------------+-------------------------------------------+
|               class               |       dscribe.descriptors.soap.SOAP       |
|               kwargs              |    {'rcut': 7.0, 'nmax': 8, 'lmax': 8}    |
|            species list           | ['Fe', 'Ni', 'Pt', 'Pd', 'Au', 'Cu', 'H'] |
|       maximum structure size      |                     37                    |
|               preset              |                    None                   |
| design space structures provided? |                    True                   |
+-----------------------------------+-------------------------------------------+
```

Similarly, we can specify the regressor to be used. The class should 
be "`sklearn`-like" with `fit` and `predict` methods.

Here we will use `sklearn`'s `GaussianProcessRegressor` for regression.
```py
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> kernel = RBF(1.5)
>>> regressor = GaussianProcessRegressor(kernel=kernel)
```

Now that we have both our `Featurizer` and regressor, we can construct 
a `Predictor` object.

```py
>>> from autocat.learning.predictors import Predictor
>>> predictor = Predictor(
...     regressor=regressor,
...     featurizer=featurizer,
... )
>>> predictor
+-----------+------------------------------------------------------------------+
|           |                            Predictor                             |
+-----------+------------------------------------------------------------------+
| regressor | <class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'> |
|  is fit?  |                              False                               |
+-----------+------------------------------------------------------------------+
+-----------------------------------+-------------------------------------------+
|                                   |                 Featurizer                |
+-----------------------------------+-------------------------------------------+
|               class               |       dscribe.descriptors.soap.SOAP       |
|               kwargs              |    {'rcut': 7.0, 'nmax': 8, 'lmax': 8}    |
|            species list           | ['Fe', 'Ni', 'Pt', 'Pd', 'Au', 'Cu', 'H'] |
|       maximum structure size      |                     37                    |
|               preset              |                    None                   |
| design space structures provided? |                    True                   |
+-----------------------------------+-------------------------------------------+
```

## Training and making predictions

With our newly defined `Predictor` we can train it using data from our 
`DesignSpace` and the `fit` method. Again, please note we are using random labels 
here, solely for demonstration purposes.

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