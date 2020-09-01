from ase import Atoms

nrr_intermediate_names = ["NNH", "NNH2", "N", "NH", "NH2", "NHNH", "NHNH2"]

# N is in atomic symbols and NH2 is in g2
nrr_mols = {
    "NNH": Atoms("N2H", [(0.0, 0.0, 0.0), (-0.2, 0.0, 1.2), (0.51, 0, 1.91)]),
    "NNH2": None,
    "NH": None,
    "NHNH": None,
    "NHNH2": None,
}
