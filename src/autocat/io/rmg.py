from ase.io import read
from rmgpy.molecule.converter import to_rdkit_mol
from rmgpy.molecule.molecule import Molecule
from rmgpy.chemkin import load_species_dictionary
from rdkit.Chem.rdmolfiles import SDWriter

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

    for phase in out_yml:
        if out_yml["Phases"][phase].get("name") == "surface":
            surfaces = out_yml["Phases"][phase]["Species"]

    rmg_surf_dict = {}
    for spec in surfaces:
        name = spec["name"]
        rmg_surf_dict[name] = {}
        rmg_surf_dict[name]["raw_dict"] = spec
        rmgmol = Molecule().from_adjacency_list
        rmg_surf_dict[name]["rmg_mol"] = rmgmol
        rmg_surf_dict[name]["ase_obj"] = rmgmol_to_ase_atoms(rmgmol)

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


def rmgmol_to_ase_atoms(rmgmol: Molecule):
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
        file_path = os.path.join(_tmp_dir, "tmp.sdf")
        writer = SDWriter(file_path)
        writer.write(rd)
        writer.close()
        ase_mol = read(file_path)
        return ase_mol
