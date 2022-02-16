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
>>> from autocat.utils import extract_structures
>>> surfs_dict = generate_surface_structures(
...     ["Pt", "Cu", "Li", "Ti"],
...     n_fixed_layers=2,
...     default_lat_param_lib="pbe_fd"
... )
>>> surfs = extract_structures(surfs_dict)
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
>>> labels = np.random.uniform(-1.5,1.5,size=len(ads_structs))
```

Taking the structures and labels we can define our `DesignSpace`.

```py
>>> from autocat.learning.sequential import DesignSpace
>>> design_space = DesignSpace(surfs, labels)
```

## Doing a single simulated sequential learning run

Given our fully explored `DesignSpace`, we can simulate a sequential learning 
search over it to gain insights into guided searches within this context.
To do this simulated run we can make use of the `simulated_sequential_learning` 
function. This will internally drive a `SequentialLearner` object which will be 
returned at the end of the run.

As before, we will need to make choices with regard to the `Predictor` settings. 
In this case we will use a `SineMatrix` featurizer alongside a `GaussianProcessRegressor`. 

We also need to select parameters with regard to candidate selection. 
This includes the acquisition function to be used,  
target window (if applicable), and number of candidates to pick at each iteration. 
Let's use a maximum uncertainty acquisition function to pick candidates based on their 
associated uncertainty values. We'll also restrict the run to conduct 5 iterations.

```py
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from dscribe import SineMatrix
>>> from autocat.learning.sequential import simulated_sequential_learning
>>> kernel = RBF(1.5)
>>> model_kwargs = {"kernel": kernel}
>>> featurization_kwargs = {
...     "design_space_structures": design_space.design_space_structures,
... }
>>> predictor_kwargs = {
...     "model_class": GaussianProcessRegressor,
...     "model_kwargs": model_kwargs,
...     "featurizer_class": SineMatrix,
...     "featurization_kwargs": featurization_kwargs
... }
>>> candidate_selection_kwargs = {"aq": "MU"}
>>> sim_seq_learn = simulated_sequential_learning(
...     full_design_space=design_space,
...     init_training_size=1,
...     number_of_sl_loops=5,
...     candidate_selection_kwargs=candidate_selection_kwargs,
...     predictor_kwargs=predictor_kwargs,
... )
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
>>> runs_history = multiple_simulated_sequential_learning_runs(
...     full_design_space=design_space,
...     init_training_size=1,
...     number_of_sl_loops=5,
...     candidate_selection_kwargs=candidate_selection_kwargs,
...     predictor_kwargs=predictor_kwargs,
...     number_of_runs=3,
...     # number_of_parallel_jobs=N if you wanted to run in parallel
... )
```

Taking the `SequentialLearner`s from within `runs_history`, their histories 
may be used to calculate more robust statistics into the simulated searches.