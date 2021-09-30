import json
import pkg_resources

__all__ = ["SEGREGATION_ENERGIES"]
"""
Values obtained from https://doi.org/10.1103/PhysRevB.59.15990

SEGREGATION_ENERGIES:
    Segregation energies for different host/dopant combinations
    For hosts used fcc: 111, bcc:110 (Fe100 also available), hcp:0001
"""

raw_seg_ener = pkg_resources.resource_filename(
    "autocat.data.segregation_energies", "segregation_energies.json"
)

with open(raw_seg_ener) as fr:
    SEGREGATION_ENERGIES = json.load(fr)
