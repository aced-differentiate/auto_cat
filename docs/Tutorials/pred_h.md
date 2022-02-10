In this tutorial we are going to show how to use the learning tools within 
AutoCat to train a regressor that can predict adsorption energies of hydrogen 
on a set of single-atom alloys.

## Creating a DesignSpace

Let's start by creating a DesignSpace. Normally each of these 
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

Finally, using both our structures and labels we can define a DesignSpace. In practice, 
if any of the labels for a structure are unknown, it can be included as a `numpy.nan` 

```py
>>> from autocat.learning.sequential import DesignSpace
>>> design_space = DesignSpace(ads_structs, labels)
```

## Training the Predictor

## Making predictions