#!/usr/bin/python

import unittest
import os
import itertools
import collections
import numpy as np
from scipy import stats
from collections import defaultdict
from numpy.testing import assert_almost_equal, assert_equal

from cutie import output, parse, utils, statistics

# ensure reproducible results for bootstrapping
np.random.seed(2)

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
                [0.80054211, 0.01666667, 0.0229774 ],
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

        self.minep_fp = os.path.dirname(os.path.realpath(__file__)) + '/n=50,alpha=0.6.csv'
        # technically improper usage of the p value file but suffices
        # for testing purposes (sample size does not match)
        with open(self.minep_fp, 'r') as f:
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
                                 [0.80054211, 0.01666667, 0.0229774 ],
                                 [0.79468572, 0.0229774 , 0.01917673]]),
            'logpvals': np.array([[-3.95405776, -0.22246615, -0.22980857],
                                  [-0.22246615, -4.0943446, -3.77324409],
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
        self.resample_k = 1
        self.fold = False
        self.fold_value = 100
        self.forward = True
        self.alpha = 0.05
        self.paired = True
        self.n_corr = self.threshold_results[0][1]
        self.pvalues = self.assign_statistics_truths['kpc']['pvalues']
        self.correlations = self.assign_statistics_truths['kpc']['correlations']
        self.mine_str = self.assign_statistics_truths['mine']['correlations']
        self.threshold = self.threshold_results[5][0]
        self.initial_corr, self.all_pairs = ([(0, 1), (1, 0), (1, 2), (2, 1)],
            [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
        self.CI_method = 'none'
        self.CI_results = {
            'log': ((-171.62134974793022, 12.116703383175945),
                    -79.75232318237714, 104.80887164658711),
            'cbrt': ((0.023205742593382067, 0.22102055617426344),
                     0.12211314938382276, 0.11283861482737229),
            'none': ((1.8032903153653718e-05, 0.013368393763513012),
                     0.006693213333333333, 0.007615386328550027)}
        self.CI_exceeds = {
            'log': ([1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]),
            'cbrt': ([0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]),
            'none': ([0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.])
        }

        # specific to pearson, paired var1 = 1 var2 = 2 with bonferonni
        self.var1, self.var2 = 1, 2
        self.sign = np.sign(self.correlations[self.var1][self.var2])
        self.pointwise_results = {
            'cutie_1pc': (np.array([0., 0., 0., 0., 0.]),
                np.array([0., 0., 0., 0., 1.]),
                np.array([0.99680549, 0.99715323, 0.9969853 , 1.        , 0.94672926]),
                np.array([0.00319451, 0.00284677, 0.0030147 , 0.        , 0.05327074])),
            'cookd': (np.array([0., 0., 0., 0., 0.]),
                np.array([0., 0., 0., 0., 1.]),
                np.array([0.09404442, 0.02992931, 0.05956787, 0.45      , 4.2525    ]),
                np.array([0.91282234, 0.97080017, 0.94325794, 0.67466001, 0.1331533 ])),
            'dffits': (np.array([0., 0., 0., 0., 0.]),
                np.array([0., 0., 0., 1., 1.]),
                np.array([-3.79941633e-01, -2.07830404e-01, -2.98360772e-01, 1.56747721e+14, -2.49615088]),
                np.array([1.26491106, 1.26491106, 1.26491106, 1.26491106, 1.26491106])),
            'dsr': (np.array([0., 0., 0., 0., 0.]),
                np.array([0., 0., 0., 1., 0.]),
                np.array([-5.49963131e-01, -4.05925012e-01, -4.91552039e-01,  2.86180876e+14, -4.44749590e-01]),
                np.array([-5.49963131e-01, -4.05925012e-01, -4.91552039e-01,  2.86180876e+14, -4.44749590e-01]))}

        self.infln_metrics = ['cutie_1pc', 'cookd', 'dffits', 'dsr']
        self.infln_mapping = {
            'cutie_1pc': statistics.resample1_cutie_pc,
            'cookd': statistics.cookd,
            'dffits': statistics.dffits,
            'dsr': statistics.dsr
        }
        self.infln_results = {
            'cutie_1pc': set(),
            'cookd': {(0, 1), (1, 0), (2, 1), (1, 2)},
            'dffits': {(0, 1), (1, 0), (2, 1), (1, 2)},
            'dsr': {(1, 2), (1, 0), (2, 1)}}

        self.complete_pointwise_results = (
            {'cutie_1pc': {(0, 1), (1, 0), (2, 1), (1, 2)},
            'cookd': {(0, 1), (1, 0), (2, 1), (1, 2)},
            'dffits': {(0, 1), (1, 0), (2, 1), (1, 2)},
            'dsr': {(1, 2), (1, 0), (2, 1)}},
            [['cutie_1pc'], ['cookd'], ['dffits'], ['dsr'], ['cutie_1pc', 'cookd'],
            ['cutie_1pc', 'dffits'], ['cutie_1pc', 'dsr'], ['cookd', 'dffits'],
            ['cookd', 'dsr'], ['dffits', 'dsr'], ['cutie_1pc', 'cookd', 'dffits'],
            ['cutie_1pc', 'cookd', 'dsr'], ['cutie_1pc', 'dffits', 'dsr'],
            ['cookd', 'dffits', 'dsr'], ['cutie_1pc', 'cookd', 'dffits', 'dsr']],
            {"['cutie_1pc']": set(), "['cookd']": set(), "['dffits']": set(),
            "['dsr']": set(), "['cutie_1pc', 'cookd']": set(),
            "['cutie_1pc', 'dffits']": set(), "['cutie_1pc', 'dsr']": set(),
            "['cookd', 'dffits']": set(), "['cookd', 'dsr']": set(),
            "['dffits', 'dsr']": set(), "['cutie_1pc', 'cookd', 'dffits']": {(0, 1)},
            "['cutie_1pc', 'cookd', 'dsr']": set(), "['cutie_1pc', 'dffits', 'dsr']": set(),
            "['cookd', 'dffits', 'dsr']": set(),
            "['cutie_1pc', 'cookd', 'dffits', 'dsr']": {(1, 2), (1, 0), (2, 1)}})

        self.update_results = {
            '[0]': (np.array([1., 0., 0., 0., 0.]),
                    np.array([0.41612579, 0.,         0.,         0.,         0.        ]),
                    np.array([-0.58387421,  1.,          1.,          1.,          1.        ])),
            '[1]': (np.array([1., 1., 0., 0., 0.]),
                    np.array([0.41612579, 0.34289777, 0.,         0.,         0.        ]),
                    np.array([-0.58387421, -0.65710223,  1.,          1.,          1.        ])),
            '[2]': (np.array([1., 1., 1., 0., 0.]),
                    np.array([0.41612579, 0.34289777, 0.3117528,  0.,         0.        ]),
                    np.array([-0.58387421, -0.65710223, -0.6882472,   1.,          1.,        ])),
            '[3]': (np.array([1., 1., 1., 1., 0.]),
                    np.array([0.41612579, 0.34289777, 0.3117528,  0.45227744, 0.        ]),
                    np.array([-0.58387421, -0.65710223, -0.6882472,  -0.54772256,  1.        ])),
            '[4]': (np.array([1., 1., 1., 1., 1.]),
                    np.array([0.41612579, 0.34289777, 0.3117528,  0.45227744, 0.48701082]),
                    np.array([-0.58387421, -0.65710223, -0.6882472, -0.54772256, -0.51298918]))}


        self.resamplek_results = {
            '1': (np.array([0., 0., 0., 0., 0.]),
                  np.array([0., 0., 0., 0., 1.]),
                  np.array([0.00319451, 0.00284677, 0.0030147 , 0.        , 0.05327074]),
                  np.array([0.99680549, 0.99715323, 0.9969853 , 1.        , 0.94672926])),
            '2': (np.array([0., 0., 0., 0., 0.]),
                  np.array([3., 3., 3., 0., 3.]),
                  np.array([0.33333333, 0.12103772, 0.21229562, 0.        , 0.33333333]),
                  np.array([0.8660254 , 0.98198051, 0.94491118, 1.        , 0.8660254 ]))}

        self.jackknifek_results = {
            '1': (np.array([0., 0., 0., 0., 0.]),
                  np.array([1., 1., 1., 1., 1.]),
                  np.array([1.        , 0.99715323, 1.        , 1.        , 1.        ]),
                  np.array([0.        , 0.00284677, 0.        , 0.        , 0.        ])),
            '2': (np.array([0., 0., 0., 0., 0.]),
                  np.array([4., 4., 4., 4., 4.]),
                  np.array([0.33333333, 0.        , 0.04785132, 0.        , 0.33333333]),
                  np.array([0.8660254 , 1.        , 0.99717646, 1.        , 0.8660254 ]))}

        self.n_replicates = 100
        self.bootstrap_results = {
            '1': (np.array([1., 1., 1., 1., 1.]),
                  np.array([1., 1., 1., 1., 1.]),
                  np.array([1., 1., 1., 1., 1.]),
                  np.array([0., 0., 0., 0., 0.])),
            '2': (np.array([1., 1., 1., 1., 1.]),
                  np.array([1., 1., 1., 1., 1.]),
                  np.array([1., 1., 1., 1., 1.]),
                  np.array([0., 0., 0., 0., 0.]))}

        self.forward_stats = ['kpc', 'jkp', 'bsp', 'ksc', 'jks', 'bss',
                     'kkc', 'jkk', 'bsk', 'mine', 'jkm', 'bsm']
        self.reverse_stats = ['rpc', 'rjkp', 'rbsp', 'rsc', 'rjks', 'rbss',
                     'rkc', 'rjkk', 'rbsk', 'rmine', 'rjkm', 'rbsm']

        self.evaluate_correlation_k_results = {
            '0': (np.array([0., 0., 0., 0., 0.]),
                  np.array([0., 0., 0., 0., 1.]),
                  0.05327073759374248, 0.9467292624062575),
            '1': (np.array([0., 0., 0., 0., 0.]),
                  np.array([3., 3., 3., 0., 3.]),
                  0.3333333333333333, 0.8660254037844387)}

        self.update_cutiek_true_corr_results = {
            '1': (defaultdict(list),
                  defaultdict(list),
                  {'1': [(0, 1), (1, 0)]},
                    {'1': np.array([[1.        , 0.32580014, 1.        ],
                                 [0.32580014, 1.        , 0.05327074],
                                 [1.        , 0.05327074, 1.        ]])},
                    {'1': np.array([[ 0.        , -0.67419986,  0.        ],
                                 [-0.67419986,  0.        ,  0.94672926],
                                 [ 0.        ,  0.94672926,  0.        ]])},
                  {'1': np.array([2., 2., 2., 2., 4.])},
                  {'1': np.array([1., 2., 1.])},
                  {'1': np.array([1., 2., 1.])},
                  {'1': {'(0, 1)': np.array([1., 1., 1., 1., 1.]),
                         '(1, 0)': np.array([1., 1., 1., 1., 1.]),
                         '(1, 2)': np.array([0., 0., 0., 0., 1.]),
                         '(2, 1)': np.array([0., 0., 0., 0., 1.])}},
                  {'1': {'(0, 1)': np.array([0., 0., 0., 0., 2.]),
                         '(1, 0)': np.array([0., 0., 0., 0., 2.]),
                         '(1, 2)': np.array([0., 0., 0., 0., 0.]),
                         '(2, 1)': np.array([0., 0., 0., 0., 0.])}})}



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
        assert_almost_equal((0.333,1), statistics.compute_kc(
            self.perfect_corr[0], self.perfect_corr[1]), decimal=3)
        assert_almost_equal((1,0.333), statistics.compute_kc(
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
                    self.alpha, mc, self.paired))
        assert_almost_equal(self.threshold_results, np.array(results), decimal=7)

    def test_get_initial_corr(self):
        assert (self.initial_corr, self.all_pairs) == statistics.get_initial_corr(
            self.n_var1, self.n_var2, self.pvalues, self.threshold, self.paired)

    # no unit test written for return_influence() because the return vars
    # are objects of the sm.OLS class

    def test_calculate_FP_sets(self):
        # True for self.fold
        assert self.infln_results == statistics.calculate_FP_sets(self.initial_corr,
            self.correlations, self.samp_var1, self.samp_var2, self.infln_metrics,
            self.infln_mapping, self.threshold, True, self.fold_value)

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


    def test_pointwise_metrics(self):
        # test cutie, cookd, dffits, dsr for just var1 = 1 var2 = 2
        influence1, influence2 = statistics.return_influence(self.var1, self.var2,
            samp_var1=self.samp_var1,samp_var2=self.samp_var1)
        for f in self.infln_mapping:
            # -7 because numbers get large
            assert_almost_equal(self.pointwise_results[f],
                self.infln_mapping[f](self.var1, self.var2, self.samp_var1,
                    self.samp_var1, influence1, influence2, self.threshold,
                    self.sign, self.fold, self.fold_value), decimal=-7)

    def test_pointwise_comparison(self):
        assert self.complete_pointwise_results == statistics.pointwise_comparison(
            self.infln_metrics, self.infln_mapping, self.samp_var1,
            self.samp_var2, self.pvalues, self.correlations, self.n_corr, self.initial_corr,
                         self.threshold, 'kpc', self.fold_value, self.paired, self.fold)


    def test_update_rev_extrema_rp(self):
        # tests updating of indicator arrays
        exceeds, reverse, extrema_p, extrema_r, var1, var2 = utils.init_var_indicators(
            self.var1, self.var2, self.samp_var1, self.samp_var2, self.forward)

        combs = [list(x) for x in itertools.combinations(range(self.n_samp), self.resample_k)]
        for indices in combs:
            # ~ operator negates the output in the case of a boolean array
            new_var1 = var1[~np.in1d(range(len(var1)), indices)]
            new_var2 = var2[~np.in1d(range(len(var2)), indices)]

            p_value, r_value = statistics.compute_pc(new_var1, new_var2)

            assert_almost_equal(self.update_results[str(indices)], statistics.update_rev_extrema_rp(
                self.sign, r_value, p_value, indices, reverse, extrema_p, extrema_r, self.forward), decimal=5)

    def test_resamplek_cutie(self):
        # test cutie
        for k in [1,2]:
            assert_almost_equal(self.resamplek_results[str(k)],
                statistics.resamplek_cutie(self.var1, self.var2, self.n_samp,
                    self.samp_var1, self.samp_var1,self.pvalues, self.threshold,
                    k, self.sign, self.forward, 'kpc', self.fold, self.fold_value,
                    self.pvalue_bins, self.mine_str, self.mine_bins))

    def test_jackknifek_cutie(self):
        # test jackknifing
        for k in [1,2]:
            assert_almost_equal(self.jackknifek_results[str(k)],
                statistics.jackknifek_cutie(self.var1, self.var2, self.n_samp,
                    self.samp_var1, self.samp_var1, self.pvalues, self.threshold,
                    k, self.sign, self.forward, 'jkp', self.CI_method,
                    self.pvalue_bins, self.mine_str, self.mine_bins))

    def test_bootstrap_cutie(self):
        # test bootstrapping
        for k in [1,2]:
            assert_almost_equal(self.bootstrap_results[str(k)],
                statistics.bootstrap_cutie(self.var1, self.var2, self.n_samp,
                    self.samp_var1, self.samp_var1, self.pvalues, self.threshold,
                    self.sign, self.forward, 'bsp', self.CI_method,
                    self.n_replicates, self.pvalue_bins, self.mine_str, self.mine_bins))


    def test_evaluation_correlation_k(self):
        # 0, 1 as opposed to 1, 2 because you add 1 insidee evaluate_correlation_k()
        for k in [0,1]:
            results = statistics.evaluate_correlation_k(
                self.var1, self.var2, self.n_samp, self.samp_var1,
                self.samp_var1, self.pvalues, self.threshold, 'kpc', k,
                self.sign, self.fold, self.fold_value, self.n_replicates,
                self.CI_method, self.forward, self.forward_stats,
                self.reverse_stats, self.pvalue_bins, self.mine_str,
                self.mine_bins)
            for r in range(len(results)):
                assert_almost_equal(results[r],
                    self.evaluate_correlation_k_results[str(k)][r])

    def test_update_cutiek_true_corr(self):
        statistic = 'kpc'
        for k in [1]:
            results = statistics.update_cutiek_true_corr(self.initial_corr,
                self.samp_var1, self.samp_var1, self.pvalues, self.correlations,
                self.threshold, self.paired, 'kpc', self.forward_stats,
                self.reverse_stats, k, self.fold, self.fold_value,
                self.n_replicates, self.CI_method, self.pvalue_bins, self.mine_bins)

            for r in range(2,len(results)):
                # if not dictionary
                if not isinstance(results[r][str(k)], collections.Mapping):
                    assert_almost_equal(results[r][str(k)],
                        self.update_cutiek_true_corr_results[str(k)][r][str(k)])
                else:
                    for key in results[r][str(k)].keys():
                        assert_almost_equal(results[r][str(k)][key],
                            self.update_cutiek_true_corr_results[str(k)][r][str(k)][key])





    def test_get_pCI(self):
        # test generation of CI for p value
        for method in ['log', 'cbrt', 'none']:
            assert(self.CI_results[method] == statistics.get_pCI(self.pvalues,
                self.n_samp,method))

    def test_test_CI(self):
        # test testing of confidence interval
        combs = [list(x) for x in itertools.combinations(range(self.n_samp), self.resample_k)]
        for method in ['log', 'cbrt', 'none']:
            exceeds = np.zeros(self.n_samp)
            # if forward is True i.e. TP/FP
            assert_almost_equal(self.CI_exceeds[method][0], statistics.test_CI(
                self.CI_results[method][0], self.threshold, exceeds,
                [item for sublist in combs for item in sublist], True, method))
            # if forward is False i.e. TN/FN
            exceeds = np.zeros(self.n_samp)
            assert_almost_equal(self.CI_exceeds[method][1], statistics.test_CI(
                self.CI_results[method][0], self.threshold, exceeds,
                [item for sublist in combs for item in sublist], False, method))


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
