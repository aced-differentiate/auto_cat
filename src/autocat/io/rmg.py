from ase.io import read as ase_read
from ase import Atoms
from rmgpy.molecule.molecule import Molecule

from rmgpy.chemkin import load_species_dictionary
from rdkit.Chem.rdmolfiles import SDWriter
from arc.species.conformers import embed_rdkit
from arc.species.conformers import rdkit_force_field
from arc.species.converter import xyz_to_xyz_file_format

from typing import List

import numpy as np

import tempfile
import os
import yaml


def output_yaml_to_surface_rmg_mol(output_yaml: str):
    """
    Reads in output yaml and collects all surface species
    into a dict of `rmgpy.Molecule` objects

    Parameters
    ----------

    output_yaml:
        String of name of yaml file from output of an `ReactionMechanismSimulator` run

    Returns
    -------

    rmg_surf_dict:
        Dict of `rmgpy.Molecule` objects  and corresponding `Atoms` objects with names as keys
        for surface species only

    """

    with open(output_yaml, "r") as f:
        out_yml = yaml.load(f)

    for phase in out_yml["Phases"]:
        if phase.get("name") == "surface":
            surfaces = phase["Species"]

    rmg_surf_dict = {}
    for spec in surfaces:
        rmgmol = Molecule().from_adjacency_list(spec["adjlist"])
        if rmgmol.is_surface_site():
            continue
        name = spec["name"]
        rmg_surf_dict[name] = {}
        rmg_surf_dict[name]["raw_dict"] = spec
        rmg_surf_dict[name]["rmg_mol"] = rmgmol
        opt = _whether_to_optimize_conformer(_remove_x(rmgmol))
        rmg_surf_dict[name]["ase_obj"] = rmgmol_to_ase_atoms_list(
            rmgmol, return_only_lowest_energy_conformer=True, optimize=opt,
        )[0]

    return rmg_surf_dict


def load_organized_species_dictionary(species_dictionary_location: str = "."):
    """
   Wrapper for `rmgpy.chemkin.load_species_dictionary` which reorganizes
   by gas and adsorbed phase species

   Parameters
   ----------

   species_dictionary_location:
       String giving path to `species_dictionary.txt` to be read

   Returns
   -------

   organized_dict:
       Dictionary of species organized by phase
   """
    dict_location = os.path.join(species_dictionary_location, "species_dictionary.txt")
    raw_dict = load_species_dictionary(dict_location)
    organized_dict = {"gas": {}, "adsorbed": {}}
    for species in raw_dict:
        if "*" in raw_dict[species].label:
            organized_dict["adsorbed"].update({species: raw_dict[species]})
        else:
            organized_dict["gas"].update({species: raw_dict[species]})
    return organized_dict


def rmgmol_to_ase_atoms_list(
    rmgmol: Molecule,
    num_conformers: int = 1,
    force_field: str = "MMFF94s",
    return_only_lowest_energy_conformer: bool = True,
    optimize: bool = True,
) -> List[Atoms]:
    """
    Converts an `rmgpy` Molecule object to a list of `ase.Atoms` object.

    Optionally optimizes the conformers using an `RDKit` MMFF forcefield.

    Parameters
    ----------

    rmgmol:
        rmg Molecule object to be converted

    num_conformers:
        Number of conformers to be generated

    force_field:
        String indicating RDKit MMFF force field to use
        Options:
        - MMFF94
        - MMFF94s

    return_only_lowest_energy_conformer:
        Bool indicating whether to return an `ase.Atoms` object for only
        the lowest energy conformer (only valid if `optimize` = True)

    optimize:
        Bool indicating whether to optimize the conformers or just directly convert to
        `ase.Atoms` objects. If set to False, will always only return a single conformer

    Returns
    -------

    aseobj:
        List of Atoms objects
    """
    rmgmol = _remove_x(rmgmol)
    rdkit_mol = embed_rdkit("_", rmgmol, num_confs=num_conformers)

    with tempfile.TemporaryDirectory() as _tmp_dir:
        if not optimize:
            return _convert_unoptimized(rdkit_mol, _tmp_dir)

        else:
            xyzs, energies = rdkit_force_field("_", rdkit_mol, force_field=force_field)
            if not energies:
                print("Warning: Unable to optimize, returning unoptimized structure")
                return _convert_unoptimized(rdkit_mol, _tmp_dir)
            if return_only_lowest_energy_conformer:
                xyzs = [xyzs[np.argmin(energies)]]
            file_path = os.path.join(_tmp_dir, "tmp.xyz")
            aseobj = []
            for xyz in xyzs:
                string = xyz_to_xyz_file_format(xyz)
                with open(file_path, "w") as f:
                    f.write(string)
                aseobj.append(ase_read(file_path))
            return aseobj


def _remove_x(rmgmol):
    """
    Removes X from rmgmol object
    """
    if rmgmol.contains_surface_site():
        x_idx = [s.symbol for s in rmgmol.atoms].index("X")
        x = rmgmol.atoms[x_idx]
        rmgmol.remove_atom(x)
        _remove_x(rmgmol)
    return rmgmol


def _whether_to_optimize_conformer(rmgmol):
    """
    If a single isolated atom, sets optimizer to False
    """
    atoms = rmgmol.atoms
    if len(atoms) > 1:
        return True
    return False


def _convert_unoptimized(rdkit_mol, _tmp_dir):
    """
    Converts an embedded RDKit Molecule directly to an ase.Atoms obj
    without optimization
    """
    file_path = os.path.join(_tmp_dir, "tmp.sdf")
    writer = SDWriter(file_path)
    writer.write(rdkit_mol)
    writer.close()
    aseobj = [ase_read(file_path)]
    return aseobj
