# AutoCat

AutoCat is a suite of python tools for **sequential learning for materials applications**
and **automating structure generation for DFT catalysis studies.**
Documentation for the package can be found
[here](https://aced-differentiate.github.io/auto_cat).

Development of this package stems from [ACED](https://www.cmu.edu/aced/), as part of the
ARPA-E DIFFERENTIATE program.

## Installation

There are two options for installation, either via `pip` or from the repo directly.

### `pip` (recommended)

If you are planning on strictly using AutoCat rather than contributing to development,
 we recommend using `pip` within a virtual environment (e.g.
 [`conda`](https://docs.conda.io/en/latest/)
 ). This can be done as follows:

```
pip install autocat
```

### Github (for developers)

Alternatively, if you would like to contribute to the development of this software,
AutoCat can be installed via a clone from Github. First, you'll need to clone the
github repo to your local machine (or wherever you'd like to use AutoCat) using
`git clone`. Once the repo has been cloned, you can install AutoCat as an editable
package by changing into the created directory (the one with `setup.py`) and installing
via:
```
pip install -e .
```

## Contributing

Contributions through issues, feature requests, and pull requests are welcome.
Guidelines are provided [here](CONTRIBUTING.md).

## Acknowledgements

The code presented herein was funded by the Advanced Research Projects Agency-Energy (ARPA-E), U.S. Department of Energy, under Award Number DE-AR0001211 and in part by the National Science Foundation, under Award Number CBET-1554273. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
