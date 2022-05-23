# Contributing

<!-- TOC -->

- [Installation and Development](#install)
    - [Running Tests](#tests)
    - [Test Coverage](#test-coverage)
- [Coding Style](#codestyle)
- [PR Submission](#pr)
- [Documentation](#documentation)

## Installation and Development<a name="install"></a>

Clone from github:
```bash
git clone https://github.com/aced-differentiate/auto_cat.git
```

Create a virtual environment;
one option is to use conda, but it is not required:
```bash
conda create -n <env_name> python=3.9
conda activate <env_name>
```

Then install requirements from within the cloned `AutoCat` directory:
```bash
pip install -U -r requirements.txt
pip install -U -r test_requirements.txt
pip install --no-deps -e .
```

### Running tests<a name="tests"></a>
We use [pytest](https://docs.pytest.org/en/stable/contents.html) to run tests.
To run all tests:
```bash
pytest -svv
```

### Test coverage<a name="test-coverage"></a>

We use [pytest-cov](https://pytest-cov.readthedocs.io/en/latest) to check
code coverage.
To run all tests and output a report of the coverage of the `src` directory:
```bash
pytest --cov=src/ --cov-report term-missing -svv
```

## Coding Style<a name="codestyle"></a>

`AutoCat` follows [PEP8](https://www.python.org/dev/peps/pep-0008/), with
several docstring rules relaxed.
See `tox.ini` for a list of the ignored rules.
Docstrings must follow the
[Numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [flake8](https://flake8.pycqa.org/en/latest/) as a linter.
To run the linter on the `src` directory:
```bash
flake8 src
```

A pre-commit hook is available to auto-format code with
[black](https://black.readthedocs.io/en/stable) (recommended):

1. Make sure you are using a Python version >=3.9
2. Install black: ``$ pip install black``
3. Install pre-commit: ``$ pip install pre-commit``
4. Intall git hooks in your ``.git`` directory: ``$ pre-commit install``

Names for functions, arguments, classes, and methods should be as descriptive as possible,
even if it means making them a little longer. For example, `generate_surface_structures` is
a preferred function name to `gen_surfs`.
All class names should adhere to [upper CamelCase](https://en.wikipedia.org/wiki/Camel_case).

## PR Submission<a name="pr"></a>

All PRs must be submitted to the `develop` branch (either fork or clone).
Releases will be from the `master` branch.

In order to be merged, a PR must be approved by one authorized user and the
build must pass.
A passing build requires the following:
* All tests pass
* The linter finds no violations of PEP8 style (not including the exceptions in `tox.ini`)
* Every line of code is executed by a test (100% coverage)
* Documentation has been updated or extended (as needed) and builds

PR descriptions should describe the motivation and context of the code changes in the PR,
both for the reviewer and also for future developers. If there's a Github issue, the PR should
be linked to the issue to provide that context.

## Documentation<a name="documentation"></a>
`AutoCat` documentation is built using `mkdocs` via
[`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/)
and
[`mkdocstrings`](https://mkdocstrings.github.io/).
All custom documentation should be written as `.md` files, appropriately placed within
`docs/`, and referenced within the `mkdocs.yml` file.

With `mkdocs` the docs webpage can be hosted locally with the command:
```bash
mkdocs serve
```
which will give an `html` link that can be pasted in a web-browser.

API documentation is automatically generated with `mkdocstrings` which parses the docstrings.
Please ensure that all docstrings follow the Numpy style.

