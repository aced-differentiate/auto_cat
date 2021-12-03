It is crucial for many heterogeneous catalysis studies to be
able to model a catalyst surface where the desired reaction
can take place. 
[`autocat.surface`](../../API/Structure_Generation/surface.md) 
provides tools for generating
low miller index surfaces for mono-element surfaces with a vacuum
in the $z$-direction.

The core function of this module is 
[`generate_surface_structures`](../../API/Structure_Generation/surface.md#autocat.surface.generate_surface_structures) 
where multiple slabs can be generated at once.

```py
>>> from autocat.surface import generate_surface_structures
>>> surf_dict = generate_surface_structures(
... ["Li", "Cu"],
... facets={"Li": ["110"]},
... supercell_dim=[5, 5, 4],
... n_fixed_layers=2,
... default_lat_param_lib="beefvdw_fd",
... write_to_disk=True,
... )
Li_bcc110 structure written to ./Li/bcc110/substrate/input.traj
Cu_fcc100 structure written to ./Cu/fcc100/substrate/input.traj
Cu_fcc111 structure written to ./Cu/fcc111/substrate/input.traj
Cu_fcc110 structure written to ./Cu/fcc110/substrate/input.traj
>>> surf_dict
{'Li': {'bcc110': {'structure': Atoms(symbols='Li100', pbc=[True, True, False], ...),
                   'traj_file_path': './Li/bcc110/substrate/input.traj'}},
 'Cu': {'fcc100': {'structure': Atoms(symbols='Cu100', pbc=[True, True, False], ...),
                   'traj_file_path': './Cu/fcc100/substrate/input.traj'},
        'fcc111': {'structure': Atoms(symbols='Cu100', pbc=[True, True, False], ...),
                   'traj_file_path': './Cu/fcc111/substrate/input.traj'},
        'fcc110': {'structure': Atoms(symbols='Cu100', pbc=[True, True, False], ...),
                   'traj_file_path': './Cu/fcc110/substrate/input.traj'}}}
```
Here we generated surface slabs for Cu and Li under the following conditions:

- for Li we only need the 110 facet
- generate all default facets for Cu
    * fcc/bcc: ["100", "110", "111"]
    * hcp: ["0001"]
- the supercell dimensions of the slabs are 5 $\times$ 5 $\times$ 4
- the bottom 2 layers are held fixed
- for structures where the lattice parameter is not explicitly specified,
their default values are pulled from the 
[`autocat.data.lattice_parameters`](../Data/lattice_parameters.md) 
library that used a BEEF-vdW XC and finite difference basis set

When using the `write_to_disk` functionality the structures
will be written into the following directory structure:

```
.
├── Cu
│   ├── fcc100
│   │   └── substrate
│   │       └── input.traj
│   ├── fcc110
│   │   └── substrate
│   │       └── input.traj
│   └── fcc111
│       └── substrate
│           └── input.traj
├── Li
│   └── bcc110
│       └── substrate
│           └── input.traj
```
