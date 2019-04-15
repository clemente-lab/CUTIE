#!/usr/bin/python

from unittest import TestCase, main

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

import stats.linregress

from cutie import output, parse, utils, statistics

class TestStatistics(TestCase):

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
        self.new_var1 = [2, 3, 4]
        self.new_var2 = [1, 1, 1]
        self.null_output = (1,0)

        self.assertEqual(self.null_output,
            statistics.compute_pc(self.new_var1, self.new_var2))

        # perfect correlation
        self.new_var1 = [2, 3, 4]
        self.new_var2 = [2, 3, 4]
        self.perfect_output = (0,1)

        self.assertEqual(self.perfect_output,
            statistics.compute_pc(self.new_var1, self.new_var2))

        #
        assert_almost_equal(cast_str_to_num(self.str1), 100000)
        assert_almost_equal(cast_str_to_num(self.str2), 0.001)


        slope, intercept, r_value, p_value, std_err = stats.linregress(
            new_var1, new_var2)

    # if p_value is nan
    if np.isnan(p_value):
        p_value = 1
        r_value = 0

    return p_value, r_value


if __name__ == '__main__':
    unittest.main()

