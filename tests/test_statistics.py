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

        self.n_var1, self.n_var2, self.n_samp = utils.get_param(self.samp_var1,
                                                                self.samp_var1)

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
        self.correlation_types = ['kpc', 'ksc', 'kkc', 'mine']

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

        self.threshold_results = np.array([
            (0.05, 6, False, 0.0002009),
            (0.05, 6, False, 0.00481823047),
            (0.05, 6, False, 0.0229774),
            (0.05, 6, False, 2.56e-07),
            (0.016666666666666666, 6, False, 0.0002009),
            (0.016666666666666666, 6, False, 0.00481823047),
            (0.016666666666666666, 6, False, 0.0229774),
            (0.016666666666666666, 6, False, 2.56e-07),
            (0.016952427508441503, 6, False, 0.0002009),
            (0.016952427508441503, 6, False, 0.00481823047),
            (0.016952427508441503, 6, False, 0.0229774),
            (0.016952427508441503, 6, False, 2.56e-07),
            (0.05, 6, False, 0.0002009),
            (0.016666666666666666, 6, False, 0.00481823047),
            (0.05, 6, True, 0.0229774),
            (0.05, 6, False, 2.56e-07)])

        # specific to pearson, bonferroni
        self.pvalues = self.assign_statistics_truths['kpc']['pvalues']
        self.correlations = self.assign_statistics_truths['kpc']['correlations']
        self.threshold = self.threshold_results[5][0]
        self.initial_corr, self.all_pairs = ([(0, 1), (1, 0), (1, 2), (2, 1)],
            [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
        self.fold_value = 100

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
        keys = ['pvalues','logpvals','correlations','r2vals']
        # tests kpc, ksc, kkc and mine
        for stat in self.correlation_types:
            stats_vals = statistics.assign_statistics(self.samp_var1,
                self.samp_var1, stat, self.pearson_stats, self.spearman_stats,
                self.kendall_stats, self.mine_stats, self.mine_bins,
                self.pvalue_bins)

            for k in range(len(keys)):
                assert_almost_equal(self.assign_statistics_truths[stat][keys[k]],
                                    stats_vals[k], decimal=7)

        # assertRaise for valuerror, 'kpp' not a valid stat string
        self.assertRaises(ValueError, statistics.assign_statistics,
            self.samp_var1, self.samp_var2, 'kpp',
            self.pearson_stats, self.spearman_stats, self.kendall_stats,
            self.mine_stats, self.mine_bins, self.pvalue_bins)

    def test_set_threshold(self):
        # test the different allowed multiple corrections adjustments
        mc_types = ['nomc', 'bc', 'fwer', 'fdr']
        results = []
        for mc in mc_types:
            for stat in self.correlation_types:
                results.append(statistics.set_threshold(
                    self.assign_statistics_truths[stat]['pvalues'],
                    0.05, mc, paired=True))
        assert_almost_equal(self.threshold_results, np.array(results), decimal=7)

    def test_get_initial_corr(self):
        assert (self.initial_corr, self.all_pairs) == statistics.get_initial_corr(
            self.n_var1, self.n_var2, self.pvalues, self.threshold, True)


    # no unit test written for return_influence() because the return vars
    # are objects of the sm.OLS class

    def test_calculate_FP_sets(self):
        infln_metrics = ['cutie_1pc', 'cookd', 'dffits', 'dsr']
        infln_mapping = {
            'cutie_1pc': statistics.resample1_cutie_pc,
            'cookd': statistics.cookd,
            'dffits': statistics.dffits,
            'dsr': statistics.dsr
        }

        # results: key is metric, entry is set of points FP to that metric
        # True signifies that fold is true
        FP_infln_sets = statistics.calculate_FP_sets(self.initial_corr,
            self.correlations, self.samp_var1, self.samp_var2, infln_metrics,
            infln_mapping, self.threshold, True, self.fold_value)

        results = {'cutie_1pc': set(),
                   'cookd': {(0, 1), (1, 0), (2, 1), (1, 2)},
                   'dffits': {(0, 1), (1, 0), (2, 1), (1, 2)},
                   'dsr': {(1, 2), (1, 0), (2, 1)}}
        assert FP_infln_sets == results

    def test_str_to_pvalues(self):
        # test str to pvalue conversion for MINE
        MIC_pvalues = np.ones(shape=[self.n_var1, self.n_var1])
        for i in range(self.n_var1):
            for j in range(self.n_var1):
                MIC_pvalues[i][j] = statistics.str_to_pvalues(self.pvalue_bins,
                    self.assign_statistics_truths['mine']['correlations'][i][j],
                    self.mine_bins)
        assert_almost_equal(self.assign_statistics_truths['mine']['pvalues'],
            MIC_pvalues)

    def test_binary_search_bins(self):
        # entry less than lowest str
        assert (False, 2) == statistics.binary_search_bins([0.9,0.8,0.7], 0.6)

        # entry in list
        assert (True, 1) == statistics.binary_search_bins([0.9,0.8,0.7], 0.85)

        # entry stronger than highest strs
        assert (True, 0) == statistics.binary_search_bins([0.9,0.8,0.7], 0.95)

    '''
        def test_pointwise_metrics(self):
            functions = [statistics.resample1_cutie_pc,
                statistics.resample1_cutie_sc,
                statistics.cookd,
                statistics.dffits,
                statistics.dsr]
            for f in functions:
                print(f)
                print(f(var1_index, var2_index, samp_var1, samp_var2,
            influence1, influence2, threshold, sign, fold, fold_value))

    def calculate_FP_sets(initial_corr, corrs, samp_var1, samp_var2, infln_metrics,
                          infln_mapping, threshold, fold, fold_value):'''

    def test_get_pCI(self):
        CI_results = {
            'log': ((-171.62134974793022, 12.116703383175945),
                    -79.75232318237714, 104.80887164658711),
            'cbrt': ((0.023205742593382067, 0.22102055617426344),
                     0.12211314938382276, 0.11283861482737229),
            'none': ((1.8032903153653718e-05, 0.013368393763513012),
                     0.006693213333333333, 0.007615386328550027)}
        for method in ['log', 'cbrt', 'none']:
            assert(CI_results[method] == statistics.get_pCI(self.pvalues,
                self.n_samp,method))


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
        # Check no zeros are returned
        samp_var = np.array([np.array([i for i in range(10)]) for j in range(10)])
        transformed = statistics.log_transform(samp_var)

        # Further checking of the values would just involve repeating code
        assert not transformed.any() == 0

        # Test all zeros
        samp_var = [[0 for i in range(10)] for j in range(10)]

        # Should raise a value error
        self.assertRaises(ValueError, statistics.log_transform, samp_var)

if __name__ == '__main__':
    unittest.main()
