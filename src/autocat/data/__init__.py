import json
import pkg_resources

__all__ = ["pbe_fd"]

"""
Calculator Settings:

    pbe_fd:
        h = 0.16, kpts = (12,12,12)

"""

raw_pbe_fd = pkg_resources.resource_filename("autocat.data", "pbe_fd.json")

with open(raw_pbe_fd) as fr:
    pbe_fd = json.load(fr)
