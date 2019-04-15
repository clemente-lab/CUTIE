#!/usr/bin/python

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from cutie import output, parse, utils, statistics

class TestStatistics(unittest.TestCase):

    def setUp(self):
        self.undef_corr = np.array([
            [2, 3, 4],
            [1, 1, 1]])
        self.perfect_corr = np.array([
            [2, 3, 4],
            [2, 3, 4]])
        self.empirical_corr = np.array([
            [2, 1, 6],
            [1, 2, 4]])

    def test_compute_pc(self):
        # undefined correlation
        assert_almost_equal((1,0), statistics.compute_pc(self.undef_corr[0],
                                                         self.undef_corr[1]))

        # perfect correlation
        assert_almost_equal((0,1), statistics.compute_pc(self.perfect_corr[0],
                                                         self.perfect_corr[1]))

        # empirical correlation
        assert_almost_equal((0.333,0.866), statistics.compute_pc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)


    def test_compute_sc(self):
        # undefined correlation
        assert_almost_equal((1,0), statistics.compute_sc(self.undef_corr[0],
                                                         self.undef_corr[1]))

        # perfect correlation
        assert_almost_equal((0,1), statistics.compute_sc(self.perfect_corr[0],
                                                         self.perfect_corr[1]))

        # empirical correlation
        assert_almost_equal((0.666,0.5), statistics.compute_sc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)



if __name__ == '__main__':
    unittest.main()

