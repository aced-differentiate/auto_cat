import os
from setuptools import setup
from setuptools import find_packages


version_file = os.path.join(os.path.dirname(__file__), "src", "autocat", "VERSION.txt")
with open(version_file, "r") as fr:
    version = fr.read().strip()

readme_file = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_file, "r") as fr:
    long_description = fr.read().strip()


setup(
    name="autocat",
    version=version,
    description="Tools for automated generation of catalyst structures and sequential learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    install_requires=[
        "numpy",
        "ase",
        "pymatgen",
        "fire",
        "matminer",
        "dscribe",
        "prettytable",
        "joblib",
    ],
    include_package_data=True,
)
