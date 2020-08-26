import unittest
import os
import shutil
from autocat import gen_saa


class TestSAA(unittest.TestCase):
    def test_CuFe(self):
        gen_saa(["Cu"], ["Fe"], supcell=(5, 5, 4), a=1.813471 * 2)


if __name__ == "__main__":
    unittest.main()
