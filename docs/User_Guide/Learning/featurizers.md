The 
[`Featurizer`](../../API/Learning/featurizers.md#autocat.learning.featurizers.Featurizer) 
object allows for the featurization of 
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
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.surface import generate_surface_structures
>>> from dscribe.descriptors import SineMatrix
>>> surfs = flatten_structures_dict(generate_surface_structures(["Li", "Na"]))
>>> f = Featurizer(SineMatrix, design_space_structures=surfs)
>>> f
+-----------------------------------+-------------------------------------------+
|                                   |                 Featurizer                |
+-----------------------------------+-------------------------------------------+
|               class               | dscribe.descriptors.sinematrix.SineMatrix |
|               kwargs              |                    None                   |
|            species list           |                ['Na', 'Li']               |
|       maximum structure size      |                     36                    |
|               preset              |                    None                   |
| design space structures provided? |                    True                   |
+-----------------------------------+-------------------------------------------+
>>> X = f.featurize_multiple(surfs)
```

```py
>>> from ase import Atoms
>>> from dscribe.descriptors import SOAP
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.surface import generate_surface_structures
>>> from autocat.adsorption import place_adsorbate
>>> surf = flatten_structures_dict(generate_surface_structures(["Cu"]))[0]
>>> ads_struct = place_adsorbate(surf, Atoms("OH"))
>>> f = Featurizer(
...    SOAP,
...    max_size=36,
...    species_list=["Cu", "O", "H"],
...    kwargs={"rcut": 6., "lmax": 8, "nmax": 8}
... )
>>> f
+-----------------------------------+-------------------------------------+
|                                   |              Featurizer             |
+-----------------------------------+-------------------------------------+
|               class               |    dscribe.descriptors.soap.SOAP    |
|               kwargs              | {'rcut': 6.0, 'lmax': 8, 'nmax': 8} |
|            species list           |           ['Cu', 'O', 'H']          |
|       maximum structure size      |                  36                 |
|               preset              |                 None                |
| design space structures provided? |                False                |
+-----------------------------------+-------------------------------------+
>>> X = f.featurize_single(ads_struct)
```

```py
>>> from autocat.learning.featurizers import Featurizer
>>> from autocat.utils import flatten_structures_dict
>>> from autocat.saa import generate_saa_structures
>>> from matminer.featurizers.composition import ElementProperty
>>> saas = flatten_structures_dict(generate_saa_structures(["Cu", "Au"],["Pt", "Pd"]))
>>> f = Featurizer(ElementProperty, preset="magpie", design_space_structures=saas)
>>> f
+-----------------------------------+------------------------------------------------------------+
|                                   |                         Featurizer                         |
+-----------------------------------+------------------------------------------------------------+
|               class               | matminer.featurizers.composition.composite.ElementProperty |
|               kwargs              |                            None                            |
|            species list           |                  ['Pt', 'Pd', 'Au', 'Cu']                  |
|       maximum structure size      |                             36                             |
|               preset              |                           magpie                           |
| design space structures provided? |                            True                            |
+-----------------------------------+------------------------------------------------------------+
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

**N.B.** `ACSF`, `SOAP`, `CrystalNNFingerprint`, `OPSiteFingerprint`, and `ChemicalSRO` 
are all implemented to featurize locally around specified 
atoms indicated with `ase.Atoms.tags <= 0`. 
The remaining implemented featurizer classes consider the full structure by definition