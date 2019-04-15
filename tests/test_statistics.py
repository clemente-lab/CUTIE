<<<<<<< HEAD
import cutie.statistics as stats
import numpy as np
from tempfile import gettempdir
from pytest import raises


def test_zero_replacement():
    # Test no zeros
    samp_var = [[i + 1 for i in range(10)] for j in range(10)]
    result = stats.zero_replacement(samp_var)
    assert result == 0.5

    # Test all zeros
    samp_var = [[0 for i in range(10)] for j in range(10)]
    # Should raise a value error
    with raises(ValueError):
        result = stats.zero_replacement(samp_var)

    # Test small numbers
    samp_var = [[float(i + 1) / 10 for i in range(10)] for j in range(10)]
    result = stats.zero_replacement(samp_var)
    assert result == (0.1 ** 2)


def test_multi_zeros():
    # Check no zeros are returned
    samp_var = np.array([np.array([i for i in range(10)]) for j in range(10)])
    samp_var_mr, samp_var_clr, samp_var_lclr, samp_var_varlog = stats.multi_zeros(samp_var)

    # Further checking of the values would just involve repeating code
    assert not samp_var_mr.any() == 0
    assert not samp_var_clr.any() == 0
    assert not samp_var_lclr.any() == 0

    # Test all zeros
    samp_var = [[0 for i in range(10)] for j in range(10)]
    # Should raise a value error
    with raises(ValueError):
        samp_var_mr, samp_var_clr, samp_var_lclr, samp_var_varlog = stats.multi_zeros(samp_var)


def test_log_transform():
    tmpdir = gettempdir() + '/'
    var_number = 3

    # Check no zeros are returned
    samp_var = np.array([np.array([i for i in range(10)]) for j in range(10)])
    transformed = stats.log_transform(samp_var, str(tmpdir), var_number)

    # Further checking of the values would just involve repeating code
    assert not transformed.any() == 0

    # Test all zeros
    samp_var = [[0 for i in range(10)] for j in range(10)]

    # Should raise a value error
    with raises(ValueError):
        transformed = stats.log_transform(samp_var, str(tmpdir), var_number)
=======
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
            [2, 3, 6],
            [1, 2, 4]])


    def test_compute_pc(self):
        # undefined correlation
        assert_almost_equal((1,0), statistics.compute_pc(self.undef_corr[0],
                                                         self.undef_corr[1]))

        # perfect correlation
        assert_almost_equal((1,0), statistics.compute_pc(self.perfect_corr[0],
                                                         self.perfect_corr[1]))

        # empirical correlation
        assert_almost_equal((0.057,0.995), statistics.compute_pc(
            self.empirical_corr[0], self.perfect_corr[1]), decimal=3)

if __name__ == '__main__':
    unittest.main()

>>>>>>> master
