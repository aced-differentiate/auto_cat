# AutoCat 

AutoCat is a suite of python tools for sequential learning with 
candidate selection metrics for materials applications. It additionally has tools 
for automated structure generation related to catalysis.

Development of this package stems from [ACED](https://www.cmu.edu/aced/), as part of the 
ARPA-E DIFFERENTIATE program.

## Installation

There are two options for installation, either via `pip` or from the repo directly.

### `pip` (recommended)

If you are planning on strictly using AutoCat rather than contributing to development,
 we recommend using `pip` within a virtual environment (e.g. 
 [`conda`](https://www.anaconda.com/products/individual)
 ). This can be done
as follows:

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