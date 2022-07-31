import unittest
import smulib

#Won't run on your pc, you'd need one additional folder (here test_data)

class Tests(unittest.TestCase):
    def test_current_sweep(self):
        smulib._sweep_measurements(".", 0.001, 0.01, 100, 2, 3.33e-5, wait=False)
    def test_data_from_file_measurements(self):
        smulib._data_from_file_measurements(["repositionned_signal_0.0020.003.csv"], "test_data", [1.5], 2, 3.33e-5, wait=False)

if __name__ == '__main__':
    Tests.test_data_from_file_measurements(Tests)