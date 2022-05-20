import json
import pkg_resources

__all__ = ["SEGREGATION_ENERGIES"]
"""

Keys:
    raban1999:
        Values obtained from https://doi.org/10.1103/PhysRevB.59.15990
        Segregation energies for different host/dopant combinations
        For hosts used fcc: 111, bcc:110 (Fe100 also available), hcp:0001

    rao2020:
        Values obtained from https://doi.org/10.1007/s11244-020-01267-2
        Segregation energies for different host/dopant combinations
"""

raw_raban_seg_ener = pkg_resources.resource_filename(
    "autocat.data.segregation_energies", "raban1999.json"
)

with open(raw_raban_seg_ener) as fr:
    RABAN1999_SEGREGATION_ENERGIES = json.load(fr)

raw_rao_seg_ener = pkg_resources.resource_filename(
    "autocat.data.segregation_energies", "rao2020.json"
)

with open(raw_rao_seg_ener) as fr:
    RAO2020_SEGREGATION_ENERGIES = json.load(fr)

SEGREGATION_ENERGIES = {
    "raban1999": RABAN1999_SEGREGATION_ENERGIES,
    "rao2020": RAO2020_SEGREGATION_ENERGIES,
}
