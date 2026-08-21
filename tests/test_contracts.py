import unittest

import numpy as np

from spectrashift.contracts import HyperspectralCube


class HyperspectralCubeTests(unittest.TestCase):
    def test_valid_cube(self) -> None:
        cube = HyperspectralCube(
            scene_id="scene-a",
            reflectance=np.ones((2, 3, 4)),
            wavelengths_nm=np.array([400, 500, 600, 700]),
            valid_mask=np.ones((2, 3), dtype=bool),
        )
        self.assertEqual(cube.bands, 4)
        self.assertEqual(cube.valid_spectra().shape, (6, 4))

    def test_rejects_unsorted_wavelengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            HyperspectralCube(
                scene_id="scene-a",
                reflectance=np.ones((2, 2, 3)),
                wavelengths_nm=np.array([400, 600, 500]),
                valid_mask=np.ones((2, 2), dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()

