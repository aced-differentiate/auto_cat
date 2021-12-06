Tools within 
[`autocat.adsorption`](../../API/Structure_Generation/adsorption.md) 
are geared towards generating structures with adsorbates placed on
a candidate catalyst surface.

The core function of this module is
[`generate_rxn_structures`](../../API/Structure_Generation/adsorption.md#autocat.adsorption.generate_rxn_structures)
 for generating multiple adsorbed structures with a single function call.

For the oxygen reduction (ORR) and nitrogen reduction (NRR) reactions,
AutoCat has default starting geometries for all of these intermediates
which can be found in [`autocat.data.intermediates`](../Data/intermediates.md). 

In addition, by default initial heights of the adsorbates are guessed based
upon the vdW radii of the nearest neighbors to the anchoring atom. 

```py
>>> from autocat.surface import generate_surface_structures
>>> from autocat.data.intermediates import ORR_INTERMEDIATE_NAMES
>>> from autocat.adsorption import generate_rxn_structures
>>> surface_dict = generate_surface_structures(
...     ["Pt"], facets={"Pt": ["111"]}, n_fixed_layers=2
... )
>>> surface = surface_dict["Pt"]["fcc111"]["structure"]
>>> ads_dict = generate_rxn_structures(
...     surface,
...     all_sym_sites=True,
...     ads=ORR_INTERMEDIATE_NAMES,
...     refs=["H2", "H2O"],
...     write_to_disk=True,
... )
OOH at (0.0,0.0) written to ./adsorbates/OOH/ontop/0.0_0.0/input.traj
OOH at (7.623,6.001) written to ./adsorbates/OOH/bridge/7.623_6.001/input.traj
OOH at (6.93,5.601) written to ./adsorbates/OOH/hollow/6.93_5.601/input.traj
OOH at (9.702,4.001) written to ./adsorbates/OOH/hollow/9.702_4.001/input.traj
O at (0.0,0.0) written to ./adsorbates/O/ontop/0.0_0.0/input.traj
O at (7.623,6.001) written to ./adsorbates/O/bridge/7.623_6.001/input.traj
O at (6.93,5.601) written to ./adsorbates/O/hollow/6.93_5.601/input.traj
O at (9.702,4.001) written to ./adsorbates/O/hollow/9.702_4.001/input.traj
OH at (0.0,0.0) written to ./adsorbates/OH/ontop/0.0_0.0/input.traj
OH at (7.623,6.001) written to ./adsorbates/OH/bridge/7.623_6.001/input.traj
OH at (6.93,5.601) written to ./adsorbates/OH/hollow/6.93_5.601/input.traj
OH at (9.702,4.001) written to ./adsorbates/OH/hollow/9.702_4.001/input.traj
H2 molecule structure written to ./references/H2/input.traj
H2O molecule structure written to ./references/H2O/input.traj
>>> ads_dict
{'OOH': {'ontop': {'0.0_0.0': {'structure': Atoms(...),
                               'traj_file_path': './adsorbates/OOH/ontop/0.0_0.0/input.traj'}},
         'bridge': {'7.623_6.001': {'structure': Atoms(...),
                                    'traj_file_path': './adsorbates/OOH/bridge/7.623_6.001/input.traj'}},
         'hollow': {'6.93_5.601': {'structure': Atoms(...),
                                   'traj_file_path': './adsorbates/OOH/hollow/6.93_5.601/input.traj'},
                    '9.702_4.001': {'structure': Atoms(...),
                                    'traj_file_path': './adsorbates/OOH/hollow/9.702_4.001/input.traj'}}},
 'O': {'ontop': {'0.0_0.0': {'structure': Atoms(...),
                             'traj_file_path': './adsorbates/O/ontop/0.0_0.0/input.traj'}},
       'bridge': {'7.623_6.001': {'structure': Atoms(...),
                                  'traj_file_path': './adsorbates/O/bridge/7.623_6.001/input.traj'}},
       'hollow': {'6.93_5.601': {'structure': Atoms(...),
                                 'traj_file_path': './adsorbates/O/hollow/6.93_5.601/input.traj'},
                  '9.702_4.001': {'structure': Atoms(...),
                                  'traj_file_path': './adsorbates/O/hollow/9.702_4.001/input.traj'}}},
 'OH': {'ontop': {'0.0_0.0': {'structure': Atoms(...),
                              'traj_file_path': './adsorbates/OH/ontop/0.0_0.0/input.traj'}},
        'bridge': {'7.623_6.001': {'structure': Atoms(...),
                                   'traj_file_path': './adsorbates/OH/bridge/7.623_6.001/input.traj'}},
        'hollow': {'6.93_5.601': {'structure': Atoms(...),
                                  'traj_file_path': './adsorbates/OH/hollow/6.93_5.601/input.traj'},
                   '9.702_4.001': {'structure': Atoms(...),
                                   'traj_file_path': './adsorbates/OH/hollow/9.702_4.001/input.traj'}}},
 'references': {'H2': {'structure': Atoms(...),
                       'traj_file_path': './references/H2/input.traj'},
                'H2O': {'structure': Atoms(...),
                        'traj_file_path': './references/H2O/input.traj'}}}
```
In the example above we are generating adsorption structures for all ORR intermediates
on all of the identified unique symmetry sites on a Pt111 slab. The unique sites are
identified using the Delaunay triangulation, as implemented in `pymatgen`. 
Additionally, by default initial heights of the adsorbates are guessed based
upon the vdW radii of the nearest neighbors to the anchoring atom.

One such use of these structures is generating free energy landscapes. 
With this in mind, there is also the `refs` parameter
which allows specifying the reference states that will be used. This generates a separate
directory containing the isolated molecule structures so that their energies may also
be calculated.

In general the dictionary generated has the following organization: 

```
{ADSORBATE_SPECIES: 
    {SITE_LABEL: 
        {XY: {"structure": Atoms, "traj_file_path": TRAJFILEPATH}}}, 
 "references": 
    {REFERENCE_SPECIES: {"structure": Atoms, "traj_file_path": TRAJFILEPATH}}}
```
When writing these adsorbed structures to disk it is done with the following subdirectory
format (mimicing the organization of the dictionary).

```
.
├── adsorbates
│   ├── O
│   │   ├── bridge
│   │   │   └── 7.623_6.001
│   │   │       └── input.traj
│   │   ├── hollow
│   │   │   ├── 6.93_5.601
│   │   │   │   └── input.traj
│   │   │   └── 9.702_4.001
│   │   │       └── input.traj
│   │   └── ontop
│   │       └── 0.0_0.0
│   │           └── input.traj
│   ├── OH
│   │   ├── bridge
│   │   │   └── 7.623_6.001
│   │   │       └── input.traj
│   │   ├── hollow
│   │   │   ├── 6.93_5.601
│   │   │   │   └── input.traj
│   │   │   └── 9.702_4.001
│   │   │       └── input.traj
│   │   └── ontop
│   │       └── 0.0_0.0
│   │           └── input.traj
│   └── OOH
│       ├── bridge
│       │   └── 7.623_6.001
│       │       └── input.traj
│       ├── hollow
│       │   ├── 6.93_5.601
│       │   │   └── input.traj
│       │   └── 9.702_4.001
│       │       └── input.traj
│       └── ontop
│           └── 0.0_0.0
│               └── input.traj
├── references
│   ├── H2
│   │   └── input.traj
│   └── H2O
│       └── input.traj
```

Instead of generating the adsorption structures for all unique sites, 
the xy coordinates of individual sites may be specified using the `sites`
 parameter.

```py
>>> from autocat.surface import generate_surface_structures
>>> from autocat.adsorption import generate_rxn_structures
>>> surface_dict = generate_surface_structures(
...     ["Pt"], facets={"Pt": ["111"]}, n_fixed_layers=2
... )
>>> surface = surface_dict["Pt"]["fcc111"]["structure"]
>>> x = surface[15].x
>>> x
4.1577878733769
>>> y = surface[15].y
>>> y
5.6011665451642
>>> site = {"custom": [(x,y)]}
>>> ads_dict = generate_rxn_structures(
...     surface,
...     ads=["Li"],
...     all_sym_sites=False,
...     sites=site,
...     write_to_disk=True,
... )
Li at (4.158,5.601) written to ./adsorbates/Li/custom/4.158_5.601/input.traj
>>> ads_dict
{'Li': {'custom': {'4.158_5.601': {'structure': Atoms(...),
                                   'traj_file_path': './adsorbates/Li/custom/4.158_5.601/input.traj'}}}}
```
