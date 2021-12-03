[`autocat.bulk`](../../API/Structure_Generation/bulk.md) 
provides tools to automatically generate mono-element
bulk structures. These are structures containing only a single
chemical species with no vacuum and 3D periodicity.

Multiple of these systems can be generated and written to
disk via a single call of 
[`generate_bulk_structures`](../../API/Structure_Generation/bulk.md#autocat.bulk.generate_bulk_structures).

``` py
>>> from autocat.bulk import generate_bulk_structures
>>> bulk_dict = generate_bulk_structures(["Pt", "Fe", "Ru"], write_to_disk=True)
Pt_bulk_fcc structure written to ./Pt_bulk_fcc/input.traj
Fe_bulk_bcc structure written to ./Fe_bulk_bcc/input.traj
Ru_bulk_hcp structure written to ./Ru_bulk_hcp/input.traj
>>> bulk_dict
{'Pt': {'crystal_structure': Atoms(symbols='Pt', pbc=True, ...),
        'traj_file_path': './Pt_bulk_fcc/input.traj'},
 'Fe': {'crystal_structure': Atoms(symbols='Fe', pbc=True, initial_magmoms=..., ...),
        'traj_file_path': './Fe_bulk_bcc/input.traj'},
 'Ru': {'crystal_structure': Atoms(symbols='Ru2', pbc=True, ...),
        'traj_file_path': './Ru_bulk_hcp/input.traj'}}
```

In general the following structure of the resulting dict is generated:

`{SPECIES: {"crystal_structure": Atoms, "traj_file_path": TRAJFILEPATH}}`

If writing to disk structures to disk via 
`#!python write_to_disk=True`,
then the following directory structure then a similar organization is maintained on the disk:

```
.
├── Fe_bulk_bcc
│   └── input.traj
├── Pt_bulk_fcc
│   └── input.traj
├── Ru_bulk_hcp
│   └── input.traj
```
where each `input.traj` contains the bulk structure.

**N.B.** by default initial magnetic moments will be set for Fe, Co, and Ni, otherwise no spin 
will be given
