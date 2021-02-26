import qml
import os
from ase import Atoms

import tempfile


def ase_atoms_to_qml_compound(ase_atoms: Atoms):
    """
    Converter from `ase.Atoms` to `qml.Compound`

    Parameters
    ----------

    ase_atoms:
        Atoms object to be converted to a qml.Compound

    Returns
    -------

    qml_compound:
        qml.Compound object corresponding to give Atoms object
    """
    with tempfile.TemporaryDirectory() as _tmp_dir:
        ase_atoms.write(os.path.join(_tmp_dir, "tmp.xyz"))
        qml_compound = qml.Compound(xyz=os.path.join(_tmp_dir, "tmp.xyz"))
        return qml_compound
