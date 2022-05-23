When characterizing a surface in the context of a 
specific reaction, calculating adsorption energies
for all of the reaction intermediates is often important.

Here, AutoCat has default structures for adsorbates 
of both the oxygen reduction reaction (ORR) and 
nitrogen reduction reaction (NRR) intermediates.

The names of all of the reaction intermediates can 
be imported and fed directly into 
AutoCat functions:
```py
>>> from autocat.data.intermediates import ORR_INTERMEDIATE_NAMES
>>> from autocat.data.intermediates import NRR_INTERMEDIATE_NAMES
>>> from autocat.surface import generate_surface_structures
>>> from autocat.utils import extract_structures
>>> from autocat.adsorption import generate_adsorbed_structures
>>> pt_dict = generate_surface_structures(["Pt"])
>>> pt_struct = extract_structures(pt_dict)[0]
>>> orr_structs = generate_adsorbed_structures(
...     surface=pt_struct, 
...     adsorbates=ORR_INTERMEDIATE_NAMES,
...     use_all_sites=True
... )
>>> nrr_structs = generate_adsorbed_structures(
...     surface=pt_struct,
...     ads=NRR_INTERMEDIATE_NAMES,
...     use_all_sites=True
... )
```
In the above example, `orr_structs` and `nrr_structs` have all of the corresponding
intermediates at every identified unique surface site.

Alternatively, if you would like to access the 
`ase.Atoms` objects for the intermediates directly,
they can be imported as a `dict`:
```py
>>> from autocat.data.intermediates import ORR_MOLS
>>> from autocat.data.intermediates import NRR_MOLS
``` 

**ORR Intermediates**: 

OOH\*, O\*, OH\*

**NRR Intermediates**: 

NNH\*, NNH$_2$\*, N\*, NH\*, NH$_2$\*, NHNH\*, NHNH$_2$\*, NH$_2$NH$_2$\*
