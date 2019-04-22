#!/usr/bin/python

import unittest
import cutie.utils as utils
import numpy as np

from tempfile import gettempdir
from pathlib import Path

class TestStatistics(unittest.TestCase):

    def test_calculate_intersection(self):
        set_names = ['Set1', 'Set2', 'Set3']
        sets = [
            {(1, 2), (3, 4)},
            {(3, 4), (4, 5)},
            {(4, 5), (1, 2)}
        ]
        r_sets, r_combs = utils.calculate_intersection(set_names, sets)
        assert r_sets["['Set1', 'Set2']"] == {(3, 4)}
        assert r_sets["['Set1', 'Set3']"] == {(1, 2)}
        assert r_sets["['Set2', 'Set3']"] == {(4, 5)}

        # Check it returns an empty list
        assert not utils.calculate_intersection([], [])[1]


    def test_get_param(self):
        # Test normal input
        n_samples = 4
        samp_var1 = [[j for j in range(15)] for i in range(n_samples)]
        samp_var2 = [[j for j in range(8)] for i in range(n_samples)]
        n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

        assert n_var1 == 15
        assert n_var2 == 8
        assert n_samp == n_samples

        # Test one file with no values
        n_samples = 4
        samp_var1 = [[j for j in range(0)] for i in range(n_samples)]
        samp_var2 = [[j for j in range(8)] for i in range(n_samples)]
        n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

        assert n_var1 == 0
        assert n_var2 == 8
        assert n_samp == n_samples

        # Test empty inputs
        n_var1, n_var2, n_samp = utils.get_param([[]], [[]])
        assert n_var1 == 0
        assert n_var2 == 0
        assert n_samp == 1


    def test_remove_nans(self):
        # Check Empty
        var1 = []
        var2 = []
        nvar1, nvar2 = utils.remove_nans(var1, var2)
        assert not nvar1.any()
        assert not nvar2.any()

        # Check some nan
        var1 = [1, np.nan, np.nan, 4, 5, 6, np.nan]
        var2 = [np.nan, 2, np.nan, 4, 5, np.nan, 7]

        nvar1, nvar2 = utils.remove_nans(var1, var2)
        assert (nvar1 == np.array([4, 5])).all()
        assert (nvar2 == np.array([4, 5])).all()

        # Check all nan
        var1 = [np.nan, np.nan, np.nan, np.nan]
        var2 = [np.nan, np.nan, np.nan, 7]

        nvar1, nvar2 = utils.remove_nans(var1, var2)
        assert not nvar1.any()
        assert not nvar2.any()

        # Check no nan
        var1 = [i for i in range(10)]
        var2 = [i * i for i in range(10)]

        nvar1, nvar2 = utils.remove_nans(var1, var2)
        assert (nvar1 == np.array(var1)).all()
        assert (nvar2 == np.array(var2)).all()


    def test_return_indicators(self):
        # Not testing the correctness of the values in the
        # list returned as they are the direct result of
        # utils.indicator and so should be tested there
        # Not tests zeros for the parameters that are only
        # passed along to utils.indicator for the same reason

        # Test a normal case
        n_var1 = 34
        n_var2 = 20
        initial_corr = {(1, 2), (2, 3), (15, 19)}
        true_corr = {
            '1': {(1, 2), (2, 3), (15, 19)},
            '2': {(2, 3)}
        }
        resample_k = 2
        assert utils.return_indicators(n_var1, n_var2, initial_corr, true_corr, resample_k)

        # Test an empty case
        assert not utils.return_indicators(0, 0, {}, {}, 0)

        # Test a partially empty cases
        assert not utils.return_indicators(n_var1, n_var2, initial_corr, true_corr, 0)

if __name__ == '__main__':
    unittest.main()
