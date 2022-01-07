In order to iterate a sequential learning pipeline,
a regressor is needed to select subsequent candidate systems.
For this purpose, there is the 
[`Predictor`](../../API/Learning/predictors.md#autocat.learning.predictors.Predictor)
object class. This contains two key attributes:

- a regressor that can be fit to data and used for predictions
(the class provided must have `fit` and `predict` methods)
- specified featurization methods that are available in 
[`autocat.learning.featurizers`](../../API/Learning/featurizers.md).
 In particular there are two currently implemented approaches,
structure methods that featurize the entire structure (e.g. `sine matrix`)
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
>>> from autocat.learning.predictors import Predictor
>>> from autocat.saa import generate_saa_structures
>>> from autocat.utils import extract_structures
>>> from sklearn.ensemble import RandomForestRegressor
>>> saa_dict = generate_saa_structures(["Cu", "Au", "Fe"], ["Pt", "Ru", "Ni"])
>>> saa_structs = extract_structures(saa_dict)
>>> labels = np.random.randint(1, size=(len(saa_structs) - 1))
>>> acp = Predictor(
...     structure_featurizer="sine_matrix", model_class=RandomForestRegressor
... )
>>> acp.fit(saa_structs[:-1], labels)
>>> pred, _ = acp.predict([saa_structs[-1]])
>>> pred
array([0.])
```
Here we have chosen to featurize the structures as a `sine matrix`.
Note as well that the `predict` method will return uncertainty estimates
if available. To see this, let's train a gaussian process regressor with an RBF
 kernel.

```py
>>> import numpy as np
>>> from autocat.learning.predictors import Predictor
>>> from autocat.saa import generate_saa_structures
>>> from autocat.utils import extract_structures
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import RBF
>>> saa_dict = generate_saa_structures(["Cu", "Au", "Fe"], ["Pt", "Ru", "Ni"])
>>> saa_structs = extract_structures(saa_dict)
>>> labels = np.random.randint(1, size=(len(saa_structs) - 1))
>>> kernel = RBF()
>>> acp = Predictor(
...     structure_featurizer="sine_matrix",
...     model_class=GaussianProcessRegressor,
...     model_kwargs={"kernel": kernel},
... )
>>> acp.fit(saa_structs[:-1], labels)
>>> pred, unc = acp.predict([saa_structs[-1]])
>>> pred
array([0.])
>>> unc
array([1.])
```
