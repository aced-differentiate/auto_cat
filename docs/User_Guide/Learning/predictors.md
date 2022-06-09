In order to iterate a sequential learning pipeline,
a regressor is needed to select subsequent candidate systems.
For this purpose, there is the 
[`Predictor`](../../API/Learning/predictors.md#autocat.learning.predictors.Predictor)
object class. This contains two key attributes:

- a regressor that can be fit to data and used for predictions
(the class provided must have `fit` and `predict` methods)
- featurizer class and kwargs to instantiate a [`Featurizer`](featurizers.md).
 In particular there are two currently implemented approaches,
structure methods that featurize the entire structure (e.g. `SineMatrix`, `ElementProperty`)
 and adsorbate methods that featurize locally (e.g. `SOAP`).

Generally, this predictor object behaves similarly to regressors found in 
[`sklearn`](https://scikit-learn.org/stable/)
with its own 
[`fit`](../../API/Learning/predictors.md#autocat.learning.predictors.Predictor.fit), 
[`predict`](../../API/Learning/predictors.md#autocat.learning.predictors.Predictor.predict), 
and 
[`score`](../../API/Learning/predictors.md#autocat.learning.predictors.Predictor.score) 
methods.

As an example, let's train a random forest regressor on some 
single atom alloys.

```py
>>> import numpy as np
>>> from dscribe.descriptors import SineMatrix
>>> from sklearn.ensemble import RandomForestRegressor
>>> from autocat.saa import generate_saa_structures
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.learning.predictors import Predictor
>>> saa_dict = generate_saa_structures(["Cu", "Au", "Fe"], ["Pt", "Ru", "Ni"])
>>> saa_structs = flatten_structures_dict(saa_dict)
>>> labels = np.random.randint(1, size=(len(saa_structs) - 1))
>>> featurizer = Featurizer(
...     featurizer_class=SineMatrix
... )
>>> regressor = RandomForestRegressor()
>>> acp = Predictor(
...     regressor=regressor,
...     featurizer=featurizer,
... )
>>> acp.fit(saa_structs[:-1], labels)
>>> acp
+-----------+----------------------------------------------------------+
|           |                        Predictor                         |
+-----------+----------------------------------------------------------+
| regressor | <class 'sklearn.ensemble._forest.RandomForestRegressor'> |
|  is fit?  |                           True                           |
+-----------+----------------------------------------------------------+
+-----------------------------------+----------------------------------------------------+
|                                   |                     Featurizer                     |
+-----------------------------------+----------------------------------------------------+
|               class               |     dscribe.descriptors.sinematrix.SineMatrix      |
|               kwargs              |                        None                        |
|            species list           | ['Fe', 'Ni', 'Pt', 'Pd', 'Cu', 'C', 'N', 'O', 'H'] |
|       maximum structure size      |                        100                         |
|               preset              |                        None                        |
| design space structures provided? |                       False                        |
+-----------------------------------+----------------------------------------------------+
>>> pred, _ = acp.predict([saa_structs[-1]])
>>> pred
array([0.])
```
Here we have chosen to featurize the structures as a `SineMatrix`.

Note as well that the `predict` method will return uncertainty estimates
if available. To see this, let's train a gaussian process regressor with an RBF
 kernel. Let's also featurize using `SOAP` to see how featurization kwargs are passed

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
>>> subs = flatten_structures_dict(generate_surface_structures(["Pt", "Fe", "Ru"]))
>>> structs = [place_adsorbate(s, Atoms("OH")) for s in subs]
>>> labels = np.random.randint(1, size=(len(structs) - 1))
>>> featurizer = Featurizer(
...     featurizer_class=SOAP,
...     design_space_structures=structs,
...     kwargs={"rcut": 6.0, "nmax": 6, "lmax": 6}
... )
>>> kernel = RBF()
>>> regressor = GaussianProcessRegressor(kernel=kernel)
>>> acp = Predictor(
...     featurizer=featurizer,
...     regressor=regressor
... )
>>> acp.fit(structs[:-1], labels)
>>> acp
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
|               kwargs              | {'rcut': 6.0, 'nmax': 6, 'lmax': 6} |
|            species list           |     ['Fe', 'Ru', 'Pt', 'O', 'H']    |
|       maximum structure size      |                  38                 |
|               preset              |                 None                |
| design space structures provided? |                 True                |
+-----------------------------------+-------------------------------------+
>>> pred, unc = acp.predict([structs[-1]])
>>> pred
array([0.])
>>> unc
array([1.])
```
