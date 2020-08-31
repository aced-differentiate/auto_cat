from ase import Atoms

orr_intermediate_names = ["OOH", "OH"]

orr_mols = {
    "OOH": Atoms("O2H", [(0.0, 0.0, 0.0), (0.0, 1.2, 0.8), (0.0, 0.8, 1.6)]),
    "OH": Atoms("OH", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.97)]),
}
