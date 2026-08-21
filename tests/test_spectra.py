import unittest

import numpy as np

from spectrashift.spectra import interpolate_wavelengths, spectral_angles


class SpectraTests(unittest.TestCase):
    def test_interpolation_identity(self) -> None:
        wavelengths = np.array([400.0, 500.0, 600.0])
        values = np.array([[0.1, 0.4, 0.2]])
        np.testing.assert_allclose(
            interpolate_wavelengths(values, wavelengths, wavelengths), values
        )

    def test_exact_reference_has_smallest_angle(self) -> None:
        cube = np.array([[[1.0, 0.0, 0.0]]])
        references = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        angles = spectral_angles(cube, references)
        self.assertLess(angles[0, 0, 0], angles[0, 0, 1])
        self.assertAlmostEqual(angles[0, 0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()

