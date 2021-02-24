from ase.io import read
from rmgpy.molecule.converter import to_rdkit_mol
from rmgpy.chemkin import load_species_dictionary
from rdkit.Chem.rdmolfiles import SDWriter

import tempfile
import os


def rmgmol_to_ase_atoms(rmgmol):
    """
    Converts an rmgpy Molecule object to an ase Atoms object

    Parameters
    ----------

    rmgmol:
        rmg Molecule object to be converted

    Returns
    -------

    aseobj:
        Atoms object
    """
    rd = to_rdkit_mol(rmgmol, remove_h=False, sanitize=False)
    with tempfile.TemporaryDirectory() as _tmp_dir:
        writer = SDWriter(os.path.join(_tmp_dir, "tmp.sdf"))
        writer.write(rd)
        writer.close()
        ase_mol = read("test.sdf")
        return ase_mol
