#!/usr/bin/python

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from cutie import output, parse, utils, statistics

class TestStatistics(unittest.TestCase):

    def setUp(self):
        self.samp_var1 = np.transpose(np.array([
            [2, 1, 3, 1, 15],
            [1, 4, 2, 3, 15],
            [1, 4, 2, 4, 15]]))
        self.samp_var2 = np.transpose(np.array([
            [3, 9, 3, 1],
            [1, 4, 2, 3],
            [2, 7, 5, 4]]))

        self.n_var1, self.n_var2, self.n_samp = utils.get_param(self.samp_var1,
            self.samp_var2)


    def test_compute_pc(self):
        # undefined correlation
        assert_almost_equal((1,0), statistics.compute_pc([2, 3, 4], [1, 1, 1]))

        # perfect correlation
        assert_almost_equal((0,1), statistics.compute_pc([2, 3, 4], [2, 3, 4]))

        # empirical correlation
        assert_almost_equal((0.057,0.995), statistics.compute_pc([2, 3, 6],[1, 2, 4]),decimal=3)

if __name__ == '__main__':
    unittest.main()

