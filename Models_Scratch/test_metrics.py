import unittest
import numpy as np
from metrics import euclidean, manhattan, minkowski

class MetricsTestCase(unittest.TestCase):
    """Tests for `metrics.py`."""

    def test_num_distance_real(self):
        """Is the distance successfully calculated?"""
        self.a = np.int(0)
        self.b = np.int(2)
        self.assertTrue(euclidean(self.a,self.b))

    def test_point_distance_real(self):
        """Is the distance successfully calculated?"""
        self.a = np.array([0])
        self.b = np.array([2])
        self.assertTrue(euclidean(self.a,self.b))

    def test_vector_distance_real(self):
        """Is the distance successfully calculated?"""
        self.a = np.array([0,0,0,0])
        self.b = np.array([2,2,2,2])
        self.assertTrue(euclidean(self.a,self.b))

# class MetricsTestCase(unittest.TestCase):
#     """Tests for `metrics.py`."""
#
#     def test_is_distance_8(self):
#         """Is the distance successfully calculated?"""
#         self.a = np.array([0,0,0,0])
#         self.b = np.array([2,2,2,2])
#         self.assertTrue(manhattan(self.a,self.b))
#
# class MetricsTestCase(unittest.TestCase):
#     """Tests for `metrics.py`."""
#
#     def test_is_distance_4(self):
#         """Is the distance successfully calculated?"""
#         self.a = np.array([0,0,0,0])
#         self.b = np.array([2,2,2,2])
#         self.assertTrue(minkowski(self.a,self.b,2))

if __name__ == '__main__':
    unittest.main()
