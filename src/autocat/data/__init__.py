import json
import pkg_resources

__all__ = ["pbe_fd", "beefvdw_fd", "pbe_pw", "beefvdw_pw"]

"""
Calculator Settings:

    pbe_fd, beefvdw_fd:

        Obtained via fits to an equation of state
        (https://wiki.fysik.dtu.dk/ase/ase/eos.html)

        FCC/BCC
        h = 0.16, kpts = (12,12,12)
        fit to an SJ EOS

        HCP
        h=0.16, kpts = (12,12,6)
        fit to a Birch-Murnaghan EOS


    pbe_pw, beefvdw_pw:

        Obtained using the Exponential Cell Filter to minimize the stress tensor and atomic forces
        (https://wiki.fysik.dtu.dk/ase/ase/constraints.html#the-expcellfilter-class)

        FCC/BCC
        mode=PW(550), kpts = (12,12,12), fmax = 0.05 eV/A

        HCP
        mode=PW(550), kpts = (12,12,6), fmax = 0.05 eV/A
 
"""

raw_pbe_fd = pkg_resources.resource_filename("autocat.data", "pbe_fd.json")

with open(raw_pbe_fd) as fr:
    pbe_fd = json.load(fr)


raw_beefvdw_fd = pkg_resources.resource_filename("autocat.data", "beefvdw_fd.json")

with open(raw_beefvdw_fd) as fr:
    beefvdw_fd = json.load(fr)


raw_pbe_pw = pkg_resources.resource_filename("autocat.data", "pbe_pw.json")

with open(raw_pbe_pw) as fr:
    pbe_pw = json.load(fr)


raw_beefvdw_pw = pkg_resources.resource_filename("autocat.data", "beefvdw_pw.json")

with open(raw_beefvdw_pw) as fr:
    beefvdw_pw = json.load(fr)
