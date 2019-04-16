#!/usr/bin/python

import unittest

import os
import numpy as np
from scipy import stats
from numpy.testing import assert_almost_equal, assert_equal

from cutie import output, parse, utils, statistics

class TestStatistics(unittest.TestCase):

    # setUp name is specific to package
    def setUp(self):
        # undefined correlation
        self.undef_corr = np.array([
            [2, 3, 4],
            [1, 1, 1]])

        # perfect correlation
        self.perfect_corr = np.array([
            [2, 3, 4],
            [2, 3, 4]])

        # empirical correlation
        self.empirical_corr = np.array([
            [2, 1, 6],
            [1, 2, 4]])

        # sample data in long/tidy format
        self.samp_var1 = np.array([[2,1,1],
                                   [1,4,4],
                                   [3,2,2],
                                   [1,3,4],
                                   [15,15,15]])

        self.samp_var2 = np.array([[3,9,3],
                                   [1,4,2],
                                   [2,7,5],
                                   [1,2,3],
                                   [3,6,2]])

        self.true_paired_sampvar1_arrays = {
            'stats.pearsonr': (np.array([
                [1.        , 0.95279653, 0.93951252],
                [0.95279653, 1.        , 0.99696214],
                [0.93951252, 0.99696214, 1.        ]]), np.array([
                [0.        , 0.0122235 , 0.01769506],
                [0.0122235 , 0.        , 0.0002009 ],
                [0.01769506, 0.0002009 , 0.        ]])),
            'stats.spearmanr': (np.array([
                [1.        , 0.15389675, 0.15789474],
                [0.15389675, 1.        , 0.97467943],
                [0.15789474, 0.97467943, 1.        ]]), np.array([
                [0.00000000e+00, 8.04828817e-01, 7.99800666e-01],
                [8.04828817e-01, 1.40426542e-24, 4.81823047e-03],
                [7.99800666e-01, 4.81823047e-03, 0.00000000e+00]])),
            'stats.kendalltau': (np.array([
                [1.        , 0.10540926, 0.11111111],
                [0.10540926, 1.        , 0.9486833 ],
                [0.11111111, 0.9486833 , 1.        ]]), np.array([
                [0.01917673, 0.80054211, 0.79468572],
                [0.80054211, 0.01430588, 0.0229774 ],
                [0.79468572, 0.0229774 , 0.01917673]])),
            'minepy': (np.array([
                [1.        , 0.41997309, 0.41997309],
                [0.41997309, 1.        , 0.97095059],
                [0.41997309, 0.97095059, 1.        ]]), np.array([
                [2.5600000e-07, 2.9577243e-02, 2.9577243e-02],
                [2.9577243e-02, 2.5600000e-07, 2.5600000e-07],
                [2.9577243e-02, 2.5600000e-07, 2.5600000e-07]]))}

        self.pearson_stats = ['kpc', 'jkp' , 'bsp', 'rpc', 'rjkp', 'rbsp']
        self.spearman_stats = ['ksc', 'jks','bss', 'rsc', 'rjks', 'rbss']
        self.kendall_stats = ['kkc', 'jkk', 'bsk', 'rkc', 'rjkk', 'rbsk']
        self.mine_stats = ['mine', 'jkm', 'bsm', 'rmine', 'rjkm', 'rbsm']

        self.functions = ['stats.pearsonr', 'stats.spearmanr', 'stats.kendalltau']
        self.mapf = {'stats.pearsonr': stats.pearsonr,
                     'stats.spearmanr': stats.spearmanr,
                     'stats.kendalltau': stats.kendalltau}

        self.minep_fp = 'n=50,alpha=0.6.csv'
        # technically improper usage of the p value file but suffices
        # for testing purposes (sample size does not match)
        with open(os.path.dirname(os.path.realpath(__file__)) + '/' + \
            self.minep_fp, 'r') as f:
                self.mine_bins, self.pvalue_bins = parse.parse_minep(f, ',', 13)

        self.assign_statistics_truths = {
            'kpc': {
            'pvalues': np.array([[0.        , 0.0122235 , 0.01769506],
                                 [0.0122235 , 0.        , 0.0002009 ],
                                 [0.01769506, 0.0002009 , 0.        ]]),
            'logpvals': np.array([[    -np.inf, -4.40439492, -4.03446983],
                                  [-4.40439492,     -np.inf, -8.51268655],
                                  [-4.03446983, -8.51268655,     -np.inf]]),
            'correlations': np.array([[1.        , 0.95279653, 0.93951252],
                                      [0.95279653, 1.        , 0.99696214],
                                      [0.93951252, 0.99696214, 1.        ]]),
            'r2vals': np.array([[1.        , 0.90782123, 0.88268377],
                                [0.90782123, 1.        , 0.99393351],
                                [0.88268377, 0.99393351, 1.        ]])},

            'ksc': {
            'pvalues': np.array([[0.00000000e+00, 8.04828817e-01, 7.99800666e-01],
                                 [8.04828817e-01, 1.40426542e-24, 4.81823047e-03],
                                 [7.99800666e-01, 4.81823047e-03, 0.00000000e+00]]),
            'logpvals': np.array([[        -np.inf,  -0.21712567,  -0.22339275],
                                  [ -0.21712567, -54.9225279 ,  -5.33534854],
                                  [ -0.22339275,  -5.33534854,         -np.inf]]),
            'correlations': np.array([[1.        , 0.15389675, 0.15789474],
                                      [0.15389675, 1.        , 0.97467943],
                                      [0.15789474, 0.97467943, 1.        ]]),
            'r2vals': np.array([[1.        , 0.02368421, 0.02493075],
                                [0.02368421, 1.        , 0.95      ],
                                [0.02493075, 0.95      , 1.        ]])},

            'kkc': {
            'pvalues': np.array([[0.01917673, 0.80054211, 0.79468572],
                                 [0.80054211, 0.01430588, 0.0229774 ],
                                 [0.79468572, 0.0229774 , 0.01917673]]),
            'logpvals': np.array([[-3.95405776, -0.22246615, -0.22980857],
                                  [-0.22246615, -4.24708475, -3.77324409],
                                  [-0.22980857, -3.77324409, -3.95405776]]),
            'correlations': np.array([[1.        , 0.10540926, 0.11111111],
                                      [0.10540926, 1.        , 0.9486833 ],
                                      [0.11111111, 0.9486833 , 1.        ]]),
            'r2vals': np.array([[1.        , 0.01111111, 0.01234568],
                                [0.01111111, 1.        , 0.9       ],
                                [0.01234568, 0.9       , 1.        ]])},

            'mine': {
            'pvalues': np.array([[2.5600000e-07, 2.9577243e-02, 2.9577243e-02],
                                 [2.9577243e-02, 2.5600000e-07, 2.5600000e-07],
                                 [2.9577243e-02, 2.5600000e-07, 2.5600000e-07]]),
            'logpvals': np.array([[-15.17808839,  -3.52075003,  -3.52075003],
                                   [ -3.52075003, -15.17808839, -15.17808839],
                                   [ -3.52075003, -15.17808839, -15.17808839]]),
            'correlations': np.array([[1.        , 0.41997309, 0.41997309],
                                      [0.41997309, 1.        , 0.97095059],
                                      [0.41997309, 0.97095059, 1.        ]]),
            'r2vals': np.array([[1.        , 0.1763774 , 0.1763774 ],
                                [0.1763774 , 1.        , 0.94274506],
                                [0.1763774 , 0.94274506, 1.        ]])}}



    def test_compute_pc(self):
        assert_almost_equal((1,0), statistics.compute_pc(self.undef_corr[0],
                                                         self.undef_corr[1]))
        assert_almost_equal((0,1), statistics.compute_pc(self.perfect_corr[0],
                                                         self.perfect_corr[1]))
        assert_almost_equal((0.333,0.866), statistics.compute_pc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)

    def test_compute_sc(self):
        assert_almost_equal((1,0), statistics.compute_sc(self.undef_corr[0],
                                                         self.undef_corr[1]))
        assert_almost_equal((0,1), statistics.compute_sc(self.perfect_corr[0],
                                                         self.perfect_corr[1]))
        assert_almost_equal((0.666,0.5), statistics.compute_sc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)

    def test_compute_kc(self):
        assert_almost_equal((1,0), statistics.compute_kc(self.undef_corr[0],
                                                         self.undef_corr[1]))
        assert_almost_equal((0.117,1), statistics.compute_kc(
            self.perfect_corr[0], self.perfect_corr[1]), decimal=3)
        assert_almost_equal((0.601,0.333), statistics.compute_kc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)


    def test_initial_stats_SLR(self):
        for stat in self.functions:
            assert_almost_equal(self.true_paired_sampvar1_arrays[stat],
                statistics.initial_stats_SLR(self.samp_var1, self.samp_var1,
                    self.mapf[stat]), decimal=7)

    def test_initial_stats_MINE(self):
        assert_almost_equal(self.true_paired_sampvar1_arrays['minepy'],
            statistics.initial_stats_MINE(np.shape(self.samp_var1)[1],
                self.samp_var1, self.mine_bins, self.pvalue_bins), decimal=7)

    def test_assign_statistics(self):
        # tests kpc, ksc, kkc and mine
        for stat in ['kpc', 'ksc', 'kkc', 'mine']:
            stat_to_matrix = statistics.assign_statistics(self.samp_var1,
                self.samp_var1, stat, self.pearson_stats, self.spearman_stats,
                self.kendall_stats, self.mine_stats, self.mine_bins,
                self.pvalue_bins)

            for k in stat_to_matrix.keys():
                assert_almost_equal(self.assign_statistics_truths[stat][k],
                                    stat_to_matrix[k], decimal=7)

        # assertRaise for valuerror, 'kpp' not a valid stat string
        self.assertRaises(ValueError, statistics.assign_statistics,
            self.samp_var1, self.samp_var2, 'kpp',
            self.pearson_stats, self.spearman_stats, self.kendall_stats,
            self.mine_stats, self.mine_bins, self.pvalue_bins)


    def test_zero_replacement(self):
        # Test no zeros
        samp_var = [[i + 1 for i in range(10)] for j in range(10)]
        result = statistics.zero_replacement(samp_var)
        assert result == 0.5

        # Test all zeros
        samp_var = [[0 for i in range(10)] for j in range(10)]
        # Should raise a value error
        self.assertRaises(ValueError, statistics.zero_replacement, samp_var)

        # Test small numbers
        samp_var = [[float(i + 1) / 10 for i in range(10)] for j in range(10)]
        result = statistics.zero_replacement(samp_var)
        assert result == (0.1 ** 2)

    def test_multi_zeros(self):
        # Check no zeros are returned
        samp_var = np.array([np.array([i for i in range(10)]) for j in range(10)])
        samp_var_mr, samp_var_clr, samp_var_lclr, samp_var_varlog = statistics.multi_zeros(samp_var)

        # Further checking of the values would just involve repeating code
        assert not samp_var_mr.any() == 0
        assert not samp_var_clr.any() == 0
        assert not samp_var_lclr.any() == 0

        # Test all zeros
        samp_var = [[0 for i in range(10)] for j in range(10)]
        # Should raise a value error
        self.assertRaises(ValueError, statistics.multi_zeros, samp_var)

    def test_log_transform(self):
         # /tests/data_processing/
        tmpdir = os.path.dirname(os.path.realpath(__file__)) + '/'
        var_number = 2

        # Check no zeros are returned
        samp_var = np.array([np.array([i for i in range(10)]) for j in range(10)])
        transformed = statistics.log_transform(samp_var, tmpdir, var_number)

        # Further checking of the values would just involve repeating code
        assert not transformed.any() == 0

        # Test all zeros
        samp_var = [[0 for i in range(10)] for j in range(10)]

        # Should raise a value error
        self.assertRaises(ValueError, statistics.log_transform, samp_var, str(tmpdir), var_number)

if __name__ == '__main__':
    unittest.main()
