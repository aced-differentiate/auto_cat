## DesignSpace

The
[`DesignSpace`](../../API/Learning/sequential.md#autocat.learning.sequential.DesignSpace) 
class object is intended to store the 
*entire* design space. As the sequential learning
loop is iterated, this can be continuously updated
with the newly found labels.

There are two key components required for this object:

1. `design_space_structures`: *all* systems to be considered as 
[`ase.Atoms`](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#module-ase.atoms) 
objects in a `list`
2. `design_space_labels`: `numpy array` of the same length as the above list
with the corresponding labels. If the label is not yet
known, set it to `numpy.nan`

**NB:** The order of the list of design space structures must
be in the same order as the labels given in the 
design space labels. 

```py
>>> import numpy as np
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.learning.sequential import DesignSpace
>>> surf_dict = generate_surface_structures(["Pt", "Pd", "Cu", "Ni"])
>>> surf_structs = flatten_structures_dict(surf_dict)
>>> labels = np.array([0.95395024, 0.63504885, np.nan, 0.08320879, np.nan,
... 0.32423194, 0.55570785, np.nan, np.nan, np.nan,
... 0.18884186, np.nan])
>>> acds = DesignSpace(surf_structs, labels)
>>> acds
+-------------------------+--------------------------+
|                         |       DesignSpace        |
+-------------------------+--------------------------+
|    total # of systems   |            12            |
| # of unlabelled systems |            6             |
|  unique species present | ['Pt', 'Pd', 'Cu', 'Ni'] |
|      maximum label      |        0.95395024        |
|      minimum label      |        0.08320879        |
+-------------------------+--------------------------+
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

## CandidateSelector

The
[`CandidateSelector`](../../API/Learning/sequential.md#autocat.learning.sequential.CandidateSelector) 
object stores information about the methodology for candidate selection, 
and can apply this to choose candidates from a design space.

Key properties specified within this object include:

1. Acquisition function to be used for calculating scores. Currently supported functions:
       - maximum likelihood of improvement (MLI)
       - maximum uncertainty (MU)
       - random 
2. Number of candidates that should be proposed for each iteration
3. Target window that the candidate should ideally fall within (this is only applicable for MLI)
4. Whether to weight each system's score by its
[HHI](../../User_Guide/Data/hhi.md) and/or 
[segregation energies](../../User_Guide/Data/segregation_energies.md)

For example, let's define a `CandidateSelector` that chooses the 3 systems based on MLI with a target 
window of between 0.25 and 0.3, and weights the scores by the HHI values.

```py
>>> from autocat.learning.sequential import CandidateSelector
>>> candidate_selector = CandidateSelector(
...    acquisition_function="MLI",
...    num_candidates_to_pick=3,
...    target_window=(0.25, 0.3),
...    include_hhi=True,
... )
>>> candidate_selector
+-------------------------------+--------------------+
|                               | Candidate Selector |
+-------------------------------+--------------------+
|      acquisition function     |        MLI         |
|    # of candidates to pick    |         3          |
|         target window         |    [0.25 0.3 ]     |
|          include hhi?         |        True        |
|            hhi type           |     production     |
| include segregation energies? |       False        |
+-------------------------------+--------------------+
```

The method `choose_candidate` applies these options to calculate the scores and propose 
the desired number of candidate systems to evaluate. A `DesignSpace` must be supplied along with 
optionally a combination of predictions and/or uncertainties depending on the acquisition function 
chosen.

Using the `DesignSpace` above, and making up some prediction and uncertainty values 
(in practice these should be from your own trained `Predictor`!), we can see how this works.

```py
>>> predictions = np.array([0.95395024, 0.63504885, 0.46160089, 0.08320879, 0.81524182,
... 0.32423194, 0.55570785, 0.75537232, 0.21824507, 0.89147292,
... 0.18884186, 0.47473003])
>>> uncertainties = np.array([0.01035017, 0.01171273, 0.00688497, 0.00514248, 0.01254998,
... 0.01047033, 0.01268476, 0.01017691, 0.01436907, 0.00878836,
... 0.00786345, 0.01341667])
>>> parent_idx, max_scores, aq_scores = candidate_selector.choose_candidate(
...    design_space=acds,
...    predictions=predictions,
...    uncertainties=uncertainties 
... )
```
Here, `parent_idx` is the indices of the proposed candidate systems in the given `DesignSpace`,
`max_scores` are the scores attributed to these identified candidates, and 
`aq_scores` are the scores for all systems.

**N.B.**: If there are `np.nan` labels within the `DesignSpace`, by default the 
candidates will be chosen exclusively from these unlabelled systems. 
Otherwise, in the case of a fully labelled `DesignSpace` the default is to consider 
all systems. However, these defaults may be overridden via the `allowed_idx` parameter.


## SequentialLearner

The 
[`SequentialLearner`](../../API/Learning/sequential.md#autocat.learning.sequential.SequentialLearner) 
object stores information regarding the latest 
iteration of the sequential learning loop including:

1. A [`Predictor`](predictors.md) (and its kwargs for both the regressor and featurizer)
2. A `CandidateSelector` for choosing the candidate systems
3. Iteration number
4. Latest `DesignSpace`
5. Candidate system(s) that is identified for the next loop.
6. Histories for predictions, uncertainties, and training indices

This object can be thought of as a central hub for the 
sequential learning workflow, with an external driver 
(either automated or manual) triggering iteration. The first 
`iterate` trains the model and identifies candidate(s) to 
start the loop.

```py
>>> import numpy as np
>>> from ase import Atoms
>>> from dscribe.descriptors import SOAP
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.adsorption import place_adsorbate
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.learning.predictors import Predictor
>>> from autocat.learning.sequential import CandidateSelector
>>> from autocat.learning.sequential import DesignSpace
>>> from autocat.learning.sequential import SequentialLearner
>>> # make the DesignSpace
>>> subs_dict = generate_surface_structures(["Pt", "Pd", "Cu", "Ni"])
>>> subs = flatten_structures_dict(subs_dict)
>>> ads_structs =[place_adsorbate(s, Atoms("Li")) for s in subs] 
>>> labels = np.array([0.95395024, 0.63504885, np.nan, 0.08320879, np.nan,
... 0.32423194, 0.55570785, np.nan, np.nan, np.nan,
... 0.18884186, np.nan])
>>> acds = DesignSpace(ads_structs, labels)
>>> # specify the featurization details
>>> featurizer = Featurizer(
...    featurizer_class=SOAP,
...    design_space_structures=acds.design_space_structures,
...    kwargs={"rcut": 5.0, "lmax": 6, "nmax": 6}       
... )
>>> # define the predictor
>>> kernel = RBF()
>>> regressor = GaussianProcessRegressor(kernel=kernel)
>>> predictor = Predictor(
...    regressor=regressor,
...    featurizer=featurizer 
... )
>>> # choose how candidates will be selected on each loop
>>> candidate_selector = CandidateSelector(
...    acquisition_function="MLI",
...    target_window=(0.1, 0.2),
...    include_hhi=True,
...    hhi_type="reserves",
...    include_segregation_energies=False
... )
>>> # set up the sequential learner
>>> acsl = SequentialLearner(
...    design_space=acds,
...    predictor=predictor,
...    candidate_selector=candidate_selector,
... )
>>> acsl.iteration_count
0
>>> acsl.iterate()
>>> acsl.iteration_count
1
>>> acsl
+----------------------------------+--------------------+
|                                  | Sequential Learner |
+----------------------------------+--------------------+
|         iteration count          |         1          |
| next candidate system structures |     ['Cu36Li']     |
|  next candidate system indices   |        [7]         |
+----------------------------------+--------------------+
+-------------------------------+--------------------+
|                               | Candidate Selector |
+-------------------------------+--------------------+
|      acquisition function     |        MLI         |
|    # of candidates to pick    |         1          |
|         target window         |     [0.1 0.2]      |
|          include hhi?         |        True        |
|            hhi type           |      reserves      |
| include segregation energies? |       False        |
+-------------------------------+--------------------+
+-------------------------+--------------------------------+
|                         |          DesignSpace           |
+-------------------------+--------------------------------+
|    total # of systems   |               12               |
| # of unlabelled systems |               6                |
|  unique species present | ['Li', 'Pt', 'Pd', 'Cu', 'Ni'] |
|      maximum label      |           0.95395024           |
|      minimum label      |           0.08320879           |
+-------------------------+--------------------------------+
+-----------+------------------------------------------------------------------+
|           |                            Predictor                             |
+-----------+------------------------------------------------------------------+
| regressor | <class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'> |
|  is fit?  |                               True                               |
+-----------+------------------------------------------------------------------+
+-----------------------------------+-------------------------------------+
|                                   |              Featurizer             |
+-----------------------------------+-------------------------------------+
|               class               |    dscribe.descriptors.soap.SOAP    |
|               kwargs              | {'rcut': 5.0, 'lmax': 6, 'nmax': 6} |
|            species list           |    ['Li', 'Ni', 'Pt', 'Pd', 'Cu']   |
|       maximum structure size      |                  37                 |
|               preset              |                 None                |
| design space structures provided? |                 True                |
+-----------------------------------+-------------------------------------+
```

## Simulated Sequential Learning

If you already have a fully explored design space and want
to simulate exploration over it, the 
[`simulated_sequential_learning`](../../API/Learning/sequential.md#autocat.learning.sequential.simulated_sequential_learning) 
function may be used.

Internally this function acts a driver on a `SequentialLearner` object, and can be 
viewed as an example for how a driver can be set up for an exploratory simulated 
sequential learning loop. As inputs it requires all parameters needed to instantiate 
a `SequentialLearner` and returns the object that has been iterated. For further analysis 
of the search, histories of the predictions, uncertainties, and the training indices for 
each iteration are kept.

```py
>>> import numpy as np
>>> from dscribe.descriptors import SineMatrix
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.learning.predictors import Predictor
>>> from autocat.learning.sequential import CandidateSelector
>>> from autocat.learning.sequential import DesignSpace
>>> from autocat.learning.sequential import simulated_sequential_learning
>>> surf_dict = generate_surface_structures(["Pt", "Pd", "Cu", "Ni"])
>>> surf_structs = flatten_structures_dict(surf_dict)
>>> labels = np.array([0.95395024, 0.63504885, 0.4567, 0.08320879, 0.87779,
... 0.32423194, 0.55570785, 0.325, 0.43616, 0.321632,
... 0.18884186, 0.1114])
>>> acds = DesignSpace(surf_structs, labels)
>>> # specify the featurization details
>>> featurizer = Featurizer(
...    featurizer_class=SineMatrix,
...    design_space_structures=acds.design_space_structures,
... )
>>> # define the predictor
>>> kernel=RBF()
>>> regressor = GaussianProcessRegressor(kernel=kernel)
>>> predictor = Predictor(
...    regressor=regressor,
...    featurizer=featurizer 
... )
>>> # choose how candidates will be selected on each loop
>>> candidate_selector = CandidateSelector(
...    acquisition_function="MLI",
...    target_window=(0.1, 0.2),
...    include_hhi=True,
...    hhi_type="reserves",
...    include_segregation_energies=False
... )
>>> # conduct the simulated sequential learning loop
>>> sim_sl = simulated_sequential_learning(
...    full_design_space=acds,
...    predictor=predictor,
...    candidate_selector=candidate_selector,
...    init_training_size=5,
...    number_of_sl_loops=3,
... )
Sequential Learning Iteration #1
Sequential Learning Iteration #2
Sequential Learning Iteration #3
```

Additionally, simulated searches are typically most useful when repeated to obtain 
statistics that are less dependent on the initialization of the design space. For this 
purpose there is the 
[`multiple_simulated_sequential_learning_runs`](../../API/Learning/sequential.md#autocat.learning.sequential.multiple_simulated_sequential_learning_runs) 
function. This returns a list of `SequentialLearner` corresponding to each individual run. Optionally, 
this function can also initiate the multiple runs across parallel processes via the 
`number_of_parallel_jobs` parameter.

```py
>>> import numpy as np
>>> from matminer.featurizers.composition import ElementProperty
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.learning.predictors import Predictor
>>> from autocat.learning.sequential import CandidateSelector
>>> from autocat.learning.sequential import DesignSpace
>>> from autocat.learning.sequential import multiple_simulated_sequential_learning_runs
>>> surf_dict = generate_surface_structures(["Pt", "Pd", "Cu", "Ni"])
>>> surf_structs = flatten_structures_dict(surf_dict)
>>> labels = np.array([0.95395024, 0.63504885, 0.4567, 0.08320879, 0.87779,
... 0.32423194, 0.55570785, 0.325, 0.43616, 0.321632,
... 0.18884186, 0.1114])
>>> acds = DesignSpace(surf_structs, labels)
>>> # specify the featurization details
>>> featurizer = Featurizer(
...    featurizer_class=ElementProperty,
...    preset="matminer",
...    design_space_structures=acds.design_space_structures,
... )
>>> # define the predictor
>>> kernel = RBF()
>>> regressor = GaussianProcessRegressor(kernel=kernel)
>>> predictor = Predictor(
...    regressor=regressor,
...    featurizer=featurizer 
... )
>>> # choose how candidates will be selected on each loop
>>> candidate_selector = CandidateSelector(
...    acquisition_function="MLI",
...    target_window=(0.1,0.2),
...    include_hhi=True,
...    hhi_type="reserves",
...    include_segregation_energies=False
... )
>>> # conduct the multiple simulated sequential learning loop
>>> multi_sim_sl = multiple_simulated_sequential_learning_runs(
...    full_design_space=acds,
...    predictor=predictor,
...    candidate_selector=candidate_selector,
...    init_training_size=5,
...    number_of_sl_loops=2,
...    number_of_runs=3,
... )
Sequential Learning Iteration #1
Sequential Learning Iteration #2
Sequential Learning Iteration #1
Sequential Learning Iteration #2
Sequential Learning Iteration #1
Sequential Learning Iteration #2
>>> len(multi_sim_sl)
3
```