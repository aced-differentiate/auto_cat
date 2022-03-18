"""Generate bulk crystal structures for 3d transition metals (except Mn) """

from autocat.bulk import generate_bulk_structures

TM3d_bulk_structures = generate_bulk_structures(
    [
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Fe",  # adds initial magnetization of 4 by default
        "Co",
        "Ni",  # adds initial magnetization of 2 by default
        "Cu",
        "Zn",
    ],
    crystal_structures=None,  # uses default lattice structures from `ase.data`
    a_dict=None,  # uses default <a> lattice parameters from `ase.data`
    c_dict=None,  # uses default <c> lattice parameters from `ase.data`
    write_to_disk=True,
    write_location=".",
)
