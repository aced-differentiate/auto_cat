import os
from setuptools import setup
from setuptools import find_packages


version_file = os.path.join(os.path.dirnam(__file__), "src", "autocat", "VERSION.txt")
with open(version_file, "r") as fr:
    version = fr.read().strip()


setup(
    name="autocat",
    version=version,
    description="Tools for automated generation of catalyst structures",
    url="https://github.com/aced-differentiate/auto_cat",
    author="Lance Kavalsky",
    author_email="lkavalsky@andrew.cmu.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "ase", "pymatgen",],
)
