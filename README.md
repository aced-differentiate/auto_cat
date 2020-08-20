# autocat
Tools for automated structure generation of catalyst systems.

Currently writes out all structures as ASE trajectory files which may be read using ASE as follows:
```python
from ase.io import read

sys = read('name_of_traj.traj')
```

## Requirements
- numpy
- ase
- pymatgen
