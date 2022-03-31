import os
from setuptools import setup
from setuptools import find_packages


version_file = os.path.join(os.path.dirname(__file__), "src", "autocat", "VERSION.txt")
with open(version_file, "r") as fr:
    version = fr.read().strip()


setup(
    name="autocat",
    version=version,
    description="Tools for automated generation of catalyst structures and sequential learning",
    url="https://github.com/aced-differentiate/auto_cat",
    author="Lance Kavalsky",
    author_email="lkavalsk@andrew.cmu.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy<=1.22.0", "ase", "pymatgen<=2022.0.17", "fire",],
    include_package_data=True,
)
