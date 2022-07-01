from ase import Atoms

ORR_INTERMEDIATE_NAMES = ["OOH", "O", "OH"]

# O is in atomic symbols
ORR_MOLS = {
    "OOH": Atoms("O2H", [(0.0, 0.0, 0.0), (0.0, 1.2, 0.8), (0.0, 0.8, 1.6)]),
    "OH": Atoms("OH", [(0.0, 0.0, 0.0), (0.69, 0.0, 0.69)]),
}
