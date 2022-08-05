In this tutorial we will show how to conduct a simulated sequential learning 
run over a fully explored design space.

## Creating a fully explored `DesignSpace`
Following a similar procedure as in the previous tutorial, we will create 
a fully explored `DesignSpace` (ie. no unknown labels). This time 
the structures will be clean mono-elemental surfaces which we can generate via 
`generate_surface_structures`.

```py
>>> # Generate the clean surfaces
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import flatten_structures_dict
>>> surfs_dict = generate_surface_structures(
...     ["Pt", "Cu", "Li", "Ti"],
...     n_fixed_layers=2,
...     default_lat_param_lib="pbe_fd"
... )
>>> surfs = flatten_structures_dict(surfs_dict)
```

In this case we specified that the default lattice parameters 
from the library calculated with the PBE XC functional and 
a finite difference basis set. 

As before, we will create random labels for all structures. But if you 
want meaningful sequential learning runs these must be actual labels relevant 
to your design space!

```py
>>> # Generate the labels for each structure
>>> import numpy as np
>>> labels = np.random.uniform(-1.5,1.5,size=len(surfs))
```

Taking the structures and labels we can define our `DesignSpace`.

```py
>>> from autocat.learning.sequential import DesignSpace
>>> design_space = DesignSpace(surfs, labels)
>>> design_space
+-------------------------+--------------------------+
|                         |       DesignSpace        |
+-------------------------+--------------------------+
|    total # of systems   |            10            |
| # of unlabelled systems |            0             |
|  unique species present | ['Pt', 'Cu', 'Li', 'Ti'] |
|      maximum label      |    1.1205404366846423    |
|      minimum label      |   -1.3259701029215702    |
+-------------------------+--------------------------+
```

## Doing a single simulated sequential learning run

Given our fully explored `DesignSpace`, we can simulate a sequential learning 
search over it to gain insights into guided searches within this context.
To do this simulated run we can make use of the `simulated_sequential_learning` 
function. This will internally drive a `SequentialLearner` object which will be 
returned at the end of the run.

As before, we will need to make choices with regard to the `Predictor` settings. 
In this case we will use a `SineMatrix` featurizer alongside a `GaussianProcessRegressor`. 

```py
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> from dscribe.descriptors.sinematrix import SineMatrix
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.learning.predictors import Predictor
>>> kernel = RBF(1.5)
>>> regressor = GaussianProcessRegressor(kernel=kernel)
>>> featurizer = Featurizer(
...     featurizer_class=SineMatrix,
...     design_space_structures=design_space.design_space_structures
... )
>>> predictor = Predictor(regressor=regressor, featurizer=featurizer)
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
|               class               | dscribe.descriptors.sinematrix.SineMatrix |
|               kwargs              |                    None                   |
|            species list           |          ['Li', 'Ti', 'Pt', 'Cu']         |
|       maximum structure size      |                     36                    |
|               preset              |                    None                   |
| design space structures provided? |                    True                   |
+-----------------------------------+-------------------------------------------+
```

We also need to select parameters with regard to candidate selection. 
This includes the acquisition function to be used,  
target window (if applicable), and number of candidates to pick at each iteration. 
This can be done via the `CandidateSelector` object.
Let's use a maximum uncertainty acquisition function to pick candidates based on their 
associated uncertainty values. 

```py
>>> from autocat.learning.sequential import CandidateSelector
>>> candidate_selector = CandidateSelector(
...     acquisition_function="MU",
...     num_candidates_to_pick=1    
... )
>>> candidate_selector
+-------------------------------+--------------------+
|                               | Candidate Selector |
+-------------------------------+--------------------+
|      acquisition function     |         MU         |
|    # of candidates to pick    |         1          |
|         target window         |        None        |
|          include hhi?         |       False        |
| include segregation energies? |       False        |
+-------------------------------+--------------------+
```

Now we have everything we need to conduct a simulated sequential learning loop. 
We'll restrict the run to conduct 5 iterations.

```py
>>> from autocat.learning.sequential import simulated_sequential_learning
>>> sim_seq_learn = simulated_sequential_learning(
...     full_design_space=design_space,
...     candidate_selector=candidate_selector,
...     predictor=predictor,
...     init_training_size=1,
...     number_of_sl_loops=5,
... )
>>> sim_seq_learn
+----------------------------------+--------------------+
|                                  | Sequential Learner |
+----------------------------------+--------------------+
|         iteration count          |         6          |
| next candidate system structures |      ['Cu36']      |
|  next candidate system indices   |        [5]         |
+----------------------------------+--------------------+
+-------------------------------+--------------------+
|                               | Candidate Selector |
+-------------------------------+--------------------+
|      acquisition function     |         MU         |
|    # of candidates to pick    |         1          |
|         target window         |        None        |
|          include hhi?         |       False        |
| include segregation energies? |       False        |
+-------------------------------+--------------------+
+-------------------------+--------------------------+
|                         |       DesignSpace        |
+-------------------------+--------------------------+
|    total # of systems   |            10            |
| # of unlabelled systems |            4             |
|  unique species present | ['Pt', 'Cu', 'Li', 'Ti'] |
|      maximum label      |    0.9712050050259604    |
|      minimum label      |   -1.3259701029215702    |
+-------------------------+--------------------------+
+-----------+------------------------------------------------------------------+
|           |                            Predictor                             |
+-----------+------------------------------------------------------------------+
| regressor | <class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'> |
|  is fit?  |                               True                               |
+-----------+------------------------------------------------------------------+
+-----------------------------------+-------------------------------------------+
|                                   |                 Featurizer                |
+-----------------------------------+-------------------------------------------+
|               class               | dscribe.descriptors.sinematrix.SineMatrix |
|               kwargs              |                    None                   |
|            species list           |          ['Li', 'Ti', 'Pt', 'Cu']         |
|       maximum structure size      |                     36                    |
|               preset              |                    None                   |
| design space structures provided? |                    True                   |
+-----------------------------------+-------------------------------------------+
```

Within the returned `SequentialLearner` object we now have information we can use 
for further analysis including prediction and uncertainty histories as well as the candidate 
selection history. 

## Doing multiple simulated sequential learning runs

It is often useful to consider the statistics of multiple independent simulated 
sequential learning runs. For this purpose we can make use of the 
`multiple_simulated_sequential_learning_runs` function. This acts in the same manner 
as for the single run verion, but will return a `SequentialLearner` object for each of the 
independent runs in a list. Moreover, the inputs remain the same except with the added option 
of running in parallel (since this is an embarrassingly parallel operation). Here we will conduct 
three independent runs in serial. 

```py
>>> from autocat.learning.sequential import multiple_simulated_sequential_learning_runs
>>> runs_history = multiple_simulated_sequential_learning_runs(
...     full_design_space=design_space,
...     candidate_selector=candidate_selector,
...     predictor=predictor,
...     init_training_size=1,
...     number_of_sl_loops=5,
...     number_of_runs=3,
...     # number_of_parallel_jobs=N if you wanted to run in parallel
... )
```

Taking the `SequentialLearner`s from within `runs_history`, their histories 
may be used to calculate more robust statistics into the simulated searches.