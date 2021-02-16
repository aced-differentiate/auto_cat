import qml

import tempfile
import os

from ase import Atoms


def get_coloumb_matrix(structure: Atoms, **kwargs):
    """
    Wrapper for `qml.mol.generate_coulomb_matrix`

    Parameters
    ----------

    structure:
        Atoms object of structure for which to generate a coulomb matrix

    Returns
    -------

    coulomb_matrix:
        Numpy array of coulomb matrix
    """
    with tempfile.TemporaryDirectory() as _tmp_dir:
        structure.write(os.path.join(_tmp_dir, "tmp.xyz"))
        mol = qml.Compound(xyz=os.path.join(_tmp_dir, "tmp.xyz"))
        mol.generate_coulomb_matrix(size=mol.natoms, **kwargs)
        return mol.representation
