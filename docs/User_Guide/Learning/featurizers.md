The `Featurizer` object allows for the featurization of 
systems into a format that can be fed into machine learning 
models. Specified within this object are all the desired 
settings for when featurizing systems. More specifically this 
includes:

- `featurizer_class`: the desired class for featurization

- `preset`: if the featurizer class can be instantiated by 
a preset, that preset can be specified here. (e.g. the `magpie` feature 
set for the `ElementProperty` featurizer class)

- `design_space_structures`: if the design space is already known, 
the structures can be specified here to extract the `max_size` and 
`species_list` parameters. supercedes `max_size` and `species_list` 
upon instantiation

- `max_size`: the largest structure size that the featurizer can 
encounter

- `species_list`: all possible species that the featurizer can 
encounter

Applying the `Featurizer` there are two main methods: 
`featurize_single` and `featurize_multiple`. The former is intended 
for featurizing a single structure. On the other hand, the latter 
can take multiple structures and returns them in a single feature 
matrix.

Below are three examples using structure, site, and compositional 
featurization methods:

```py
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.utils import extract_structures
>>> from autocat.surface import generate_surface_structures
>>> from dscribe.descriptors import SineMatrix
>>> surfs = extract_structures(generate_surface_structures(["Li", "Na"]))
>>> f = Featurizer(SineMatrix, design_space_structures=surfs)
>>> f.max_size
36
>>> f.species_list
['Li', 'Na']
>>> X = f.featurize_multiple(surfs)
```

```py
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.utils import extract_structures
>>> from autocat.surface import generate_surface_structures
>>> from autocat.adsorption import place_adsorbate
>>> from dscribe.descriptors import SOAP
>>> surf = extract_structures(generate_surface_structures(["Cu"]))[0]
>>> ads_struct = extract_structures(place_adsorbate(surf, "OH", position=(0.0, 0.0)))[0]
>>> f = Featurizer(
...    SOAP,
...    max_size=36,
...    species_list=["Cu", "O", "H"]
...    kwargs={"rcut": 6., "lmax": 8, "nmax": 8}
... )
>>> X = f.featurize_single(ads_struct)
```

```py
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.utils import extract_structures
>>> from autocat.surface import generate_saa_structures
>>> from matminer.featurizers.composition import ElementProperty
>>> saas = extract_structures(generate_saa_structures(["Cu", "Au"],["Pt", "Pd"]))
>>> f = Featurizer(ElementProperty, preset="magpie", design_space_structures=saas)
>>> f.species_list
['Cu', 'Pt', 'Pd', 'Au']
>>> X = f.featurize_multiple(saas)
```

The goal of this `Featurizer` object is to provide a unified class across different 
featurization techniques.

At present the following featurizer classes are supported:

- [`dscribe`](https://singroup.github.io/dscribe/latest/):
    - `SineMatrix`
    - `CoulombMatrix`
    - `ACSF`
    - `SOAP`

- [`matminer`](https://hackingmaterials.lbl.gov/matminer/):
    - `ElementProperty`
    - `ChemicalSRO`
    - `OPSiteFingerprint`
    - `CrystalNNFingerprint`