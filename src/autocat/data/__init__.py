import json
import pkg_resources

__all__ = ["pbe_fd", "pbe_pw"]

"""
Calculator Settings:

    pbe_fd:

        Obtained via fits to an equation of state
        (https://wiki.fysik.dtu.dk/ase/ase/eos.html)

        FCC/BCC
        h = 0.16, kpts = (12,12,12)
        fit to an SJ EOS

        HCP
        h=0.16, kpts = (12,12,6)
        fit to a Birch-Murnaghan EOS


    pbe_pw:

        Obtained using the Exponential Cell Filter to minimize the stress tensor and atomic forces
        (https://wiki.fysik.dtu.dk/ase/ase/constraints.html#the-expcellfilter-class)

        FCC/BCC
        mode=PW(550), kpts = (12,12,12)

        HCP
        mode=PW(550), kpts = (12,12,6)
"""

raw_pbe_fd = pkg_resources.resource_filename("autocat.data", "pbe_fd.json")

with open(raw_pbe_fd) as fr:
    pbe_fd = json.load(fr)


raw_pbe_pw = pkg_resources.resource_filename("autocat.data", "pbe_pw.json")

with open(raw_pbe_pw) as fr:
    pbe_pw = json.load(fr)
