## DesignSpace

The
[`DesignSpace`](../../API/Learning/sequential.md#autocat.learning.sequential.DesignSpace) 
class object is intended to store the 
*entire* design space. As the sequential learning
loop is iterated, this can be continuously updated
with the newly found labels.

There are two key components required for this object:

1. `design_space_structures`: *all* systems to be considered as `ase.Atoms` objects
in a `list`
2. `design_space_labels`: `numpy array` of the same length as the above list
with the corresponding labels. If the label is not yet
known, set it to `numpy.nan`

**NB:** The order of the list of design space structures must
be in the same order as the labels given in the 
design space labels. 

```py
>>> import numpy as np
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import extract_structures
>>> from autocat.learning.sequential import DesignSpace
>>> surf_dict = generate_surface_structures(["Pt", "Pd", "Cu", "Ni"])
>>> surf_structs = extract_structures(surf_dict)
>>> labels = np.array([0.95395024, 0.63504885, np.nan, 0.08320879, np.nan,
... 0.32423194, 0.55570785, np.nan, np.nan, np.nan,
... 0.18884186, np.nan])
>>> acds = DesignSpace(surf_structs, labels)
>>> len(acds)
12
>>> acds.design_space_structures
[Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...),
 Atoms(...)]
>>> acds.design_space_labels
array([0.95395024, 0.63504885,        nan, 0.08320879,        nan,
       0.32423194, 0.55570785,        nan,        nan,        nan,
       0.18884186,        nan])
```


## SequentialLearner

The 
[`SequentialLearner`](../../API/Learning/sequential.md#autocat.learning.sequential.SequentialLearner) 
object stores information regarding the latest 
iteration of the sequential learning loop including:

1. A [`Predictor`](../../API/Learning/predictors.md#autocat.learning.predictors.Predictor)
2. Candidate selection kwargs for score calculation (e.g. acquisition functions)
3. Iteration number
4. Latest `DesignSpace`
5. Candidate system that is identified for the next loop.

This object can be thought of as a central hub for the 
sequential learning workflow, with an external driver 
(either automated or manual) triggering iteration.

```py
>>> import numpy as np
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import extract_structures
>>> from autocat.learning.sequential import DesignSpace
>>> from autocat.learning.sequential import SequentialLearner
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> surf_dict = generate_surface_structures(["Pt", "Pd", "Cu", "Ni"])
>>> surf_structs = extract_structures(surf_dict)
>>> labels = np.array([0.95395024, 0.63504885, np.nan, 0.08320879, np.nan,
... 0.32423194, 0.55570785, np.nan, np.nan, np.nan,
... 0.18884186, np.nan])
>>> acds = DesignSpace(surf_structs, labels)
>>> kernel = RBF()
>>> acsl = SequentialLearner(
...    acds,
...    predictor_kwargs={
...        "structure_featurizer": "sine_matrix",
...        "model_class": GaussianProcessRegressor,
...        "model_kwargs": {"kernel": kernel},
...    },
...    candidate_selection_kwargs={
...        "aq": "MLI",
...        "target_min": -2.25,
...        "target_max": -1.5,
...        "include_hhi": True,
...        "hhi_type": "reserves",
...        "include_seg_ener": False,
...    },
... )
>>> acsl.iteration_count
0
```
