In some codes, optimizing cell parameters on the fly
during geometry relaxations is not available.
For this reason we have compiled 
calculated lattice parameters
using multiple different
calculation schemes as a convenience for high-throughput
studies. Every calculation was conducted with
[`GPAW`](https://wiki.fysik.dtu.dk/gpaw/index.html).

There are two axes to the settings applied here:

- exchange-correlation functional
- basis set mode (finite difference or plane-wave).

Available sets are as follows:

- `BULK_PBE_FD`/`BULK_BEEFVDW_FD`: 
```
These are parameters using the finite difference scheme
and PBE / BEEF-vdW XC functionals. Obtained via fits to an 
equation of state (https://wiki.fysik.dtu.dk/ase/ase/eos.html)

FCC/BCC
h = 0.16, kpts = (12,12,12)
fit to an SJ EOS

HCP
h=0.16, kpts = (12,12,6)
fit to a Birch-Murnaghan EO
```
- `BULK_PBE_PW`/`BULK_BEEFVDW_PW`:
```
These are parameters are obatined with a plane-wave basis set and 
using the Exponential Cell Filter to minimize the stress tensor and atomic forces
(https://wiki.fysik.dtu.dk/ase/ase/constraints.html#the-expcellfilter-class)

FCC/BCC
mode=PW(550), kpts = (12,12,12), fmax = 0.05 eV/A

HCP
mode=PW(550), kpts = (12,12,6), fmax = 0.05 eV/A
```

All of these lattice parameters are available within `autocat.data.lattice_parameters`
