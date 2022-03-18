import json
import pkg_resources

__all__ = ["BULK_PBE_FD", "BULK_BEEFVDW_FD", "BULK_PBE_PW", "BULK_BEEFVDW_PW"]
"""
Calculator Settings:

    BULK_PBE_FD, BULK_BEEFVDW_FD:

        Obtained via fits to an equation of state
        (https://wiki.fysik.dtu.dk/ase/ase/eos.html)

        FCC/BCC
        h = 0.16, kpts = (12,12,12)
        fit to an SJ EOS

        HCP
        h=0.16, kpts = (12,12,6)
        fit to a Birch-Murnaghan EO

    BULK_PBE_PW, BULK_BEEFVDW_PW:

        Obtained using the Exponential Cell Filter to minimize the stress tensor and atomic forces
        (https://wiki.fysik.dtu.dk/ase/ase/constraints.html#the-expcellfilter-class)

        FCC/BCC
        mode=PW(550), kpts = (12,12,12), fmax = 0.05 eV/A

        HCP
        mode=PW(550), kpts = (12,12,6), fmax = 0.05 eV/A

"""

raw_bulk_pbe_fd = pkg_resources.resource_filename(
    "autocat.data.lattice_parameters", "bulk_pbe_fd.json"
)

with open(raw_bulk_pbe_fd) as fr:
    BULK_PBE_FD = json.load(fr)


raw_bulk_beefvdw_fd = pkg_resources.resource_filename(
    "autocat.data.lattice_parameters", "bulk_beefvdw_fd.json"
)

with open(raw_bulk_beefvdw_fd) as fr:
    BULK_BEEFVDW_FD = json.load(fr)


raw_bulk_pbe_pw = pkg_resources.resource_filename(
    "autocat.data.lattice_parameters", "bulk_pbe_pw.json"
)

with open(raw_bulk_pbe_pw) as fr:
    BULK_PBE_PW = json.load(fr)


raw_bulk_beefvdw_pw = pkg_resources.resource_filename(
    "autocat.data.lattice_parameters", "bulk_beefvdw_pw.json"
)

with open(raw_bulk_beefvdw_pw) as fr:
    BULK_BEEFVDW_PW = json.load(fr)
