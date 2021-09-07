import json
import pkg_resources

__all__ = ["HHI_PRODUCTION", "HHI_RESERVES"]
"""
Values obtained from dx.doi.org/10.1021/cm400893e

HHI_PRODUCTION:
    Calculated based on elemental production

HHI_RESERVES:
    Calculated based on known elemental reserves
"""

raw_hhi_p = pkg_resources.resource_filename("autocat.data.hhi", "hhi_p.json")

with open(raw_hhi_p) as fr:
    HHI_PRODUCTION = json.load(fr)

raw_hhi_r = pkg_resources.resource_filename("autocat.data.hhi", "hhi_r.json")

with open(raw_hhi_r) as fr:
    HHI_RESERVES = json.load(fr)
