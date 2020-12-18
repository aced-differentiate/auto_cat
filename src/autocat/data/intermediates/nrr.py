from ase import Atoms

NRR_INTERMEDIATE_NAMES = ["NNH", "NNH2", "N", "NH", "NH2", "NHNH", "NHNH2", "NH2NH2"]

# N is in atomic symbols and NH2 is in g2
NRR_MOLS = {
    "NNH": Atoms("N2H", [(0.0, 0.0, 0.0), (-0.2, 0.0, 1.2), (0.51, 0, 1.91)]),
    "NH": Atoms("NH", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
    "NH2": Atoms(
        "NH2",
        [
            [0.0, 0.0, 0.0],
            [8.31e-01, -1.76e-01, 5.48e-01],
            [-8.12e-01, -1.74e-01, 5.76e-01],
        ],
    ),
    "NNH2": Atoms(
        "N2H2",
        [
            [0.0, 0.0, 0.0],
            [-5.49e-02, -7.36e-02, 1.28],
            [-9.03e-01, -3.29e-01, 1.78],
            [7.52e-01, 1.67e-01, 1.85],
        ],
    ),
    "NH": Atoms("NH", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
    "NHNH": Atoms(
        "N2H2",
        [
            [0.0, 0.0, 0.0],
            [-7.75e-01, 7.06e-02, 1.02e00],
            [-1.72e00, 2.40e-01, 6.55e-01],
            [9.53e-01, -1.72e-01, 3.43e-01],
        ],
    ),
    "NHNH2": Atoms(
        "N2H3",
        [
            [0.0, 0.0, 0.0],
            [-8.95e-01, 4.90e-02, 1.10],
            [-1.79, 3.73e-01, 7.49e-01],
            [8.90e-01, -3.29e-01, 3.64e-01],
            [-5.81e-01, 7.38e-01, 1.78],
        ],
    ),
    "NH2NH2": Atoms(
        "N2H4",
        [
            [0.0, 0.0, 0.0],
            [-6.64e-01, 6.76e-01, 1.13],
            [-1.56, 9.64e-01, 7.48e-01],
            [9.40e-01, -2.20e-01, 3.22e-01],
            [-1.58e-01, 1.56e00, 1.24],
            [-4.63e-01, -9.04e-01, -7.71e-02],
        ],
    ),
}
