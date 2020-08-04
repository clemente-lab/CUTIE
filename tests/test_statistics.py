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
                [1.        , np.nan    , np.nan    ],
                [0.95279653, 1.        , np.nan    ],
                [0.93951252, 0.99696214, 1.        ]]), np.array([
                [0.        , np.nan    , np.nan    ],
                [0.0122235 , 0.        , np.nan    ],
                [0.01769506, 0.0002009 , 0.        ]])),
            'stats.spearmanr': (np.array([
                [1.        , np.nan    , np.nan    ],
                [0.15389675, 1.        , np.nan    ],
                [0.15789474, 0.97467943, 1.        ]]), np.array([
                [0.00000000e+00,         np.nan,         np.nan],
                [8.04828817e-01, 0.00000000e+00,         np.nan],
                [7.99800666e-01, 4.81823047e-03, 0.00000000e+00]])),
            'stats.kendalltau': (np.array([
                [1.        , np.nan    , np.nan    ],
                [0.10540926, 1.        , np.nan    ],
                [0.11111111, 0.9486833 , 1.        ]]), np.array([
                [0.        , np.nan    , np.nan    ],
                [0.80054211, 0.        , np.nan    ],
                [0.79468572, 0.0229774 , 0.        ]]))}

        self.pearson_stats = ['pearson', 'rpearson']
        self.spearman_stats = ['spearman', 'rspearman']
        self.kendall_stats = ['kendall', 'rkendall']
        self.correlation_types = ['pearson', 'spearman', 'kendall']

        self.functions = ['stats.pearsonr', 'stats.spearmanr', 'stats.kendalltau']
        self.mapf = {'stats.pearsonr': stats.pearsonr,
                     'stats.spearmanr': stats.spearmanr,
                     'stats.kendalltau': stats.kendalltau}

        self.assign_statistics_truths = {
            'pearson': {
                'pvalues': np.array([[0.        , np.nan    , np.nan    ],
                                     [0.0122235 , 0.        , np.nan    ],
                                     [0.01769506, 0.0002009 , 0.        ]]),
                'correlations': np.array([[1.        , np.nan    , np.nan    ],
                                          [0.95279653, 1.        , np.nan    ],
                                          [0.93951252, 0.99696214, 1.        ]]),
                'r2vals': np.array([[1.        , np.nan    , np.nan    ],
                                    [0.90782123, 1.        , np.nan    ],
                                    [0.88268377, 0.99393351, 1.        ]])},

            'spearman': {
                'pvalues': np.array([[0.00000000e+00, np.nan        , np.nan        ],
                                     [8.04828817e-01, 0.00000000e+00, np.nan        ],
                                     [7.99800666e-01, 4.81823047e-03, 0.00000000e+00]]),
                'correlations': np.array([[1.        , np.nan    , np.nan    ],
                                          [0.15389675, 1.        , np.nan    ],
                                          [0.15789474, 0.97467943, 1.        ]]),
                'r2vals': np.array([[1.        , np.nan    , np.nan    ],
                                    [0.02368421, 1.        , np.nan    ],
                                    [0.02493075, 0.95      , 1.        ]])},

            'kendall': {
                'pvalues': np.array([[0.        , np.nan    , np.nan    ],
                                     [0.80054211, 0.        , np.nan    ],
                                     [0.79468572, 0.0229774 , 0.        ]]),
                'correlations': np.array([[1.        , np.nan    , np.nan    ],
                                          [0.10540926, 1.        , np.nan    ],
                                          [0.11111111, 0.9486833 , 1.        ]]),
                'r2vals': np.array([[1.        , np.nan    , np.nan    ],
                                    [0.01111111, 1.        , np.nan    ],
                                    [0.01234568, 0.9       , 1.        ]])}}

        self.threshold_results = {
            'p': np.array([
                 [5.00000000e-02, 3.00000000e+00, 2.00900000e-04],
                 [5.00000000e-02, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-02, 3.00000000e+00, 2.29774000e-02],
                 [1.66666667e-02, 3.00000000e+00, 2.00900000e-04],
                 [1.66666667e-02, 3.00000000e+00, 4.81823047e-03],
                 [1.66666667e-02, 3.00000000e+00, 2.29774000e-02],
                 [1.69524275e-02, 3.00000000e+00, 2.00900000e-04],
                 [1.69524275e-02, 3.00000000e+00, 4.81823047e-03],
                 [1.69524275e-02, 3.00000000e+00, 2.29774000e-02],
                 [5.00000000e-02, 3.00000000e+00, 2.00900000e-04],
                 [1.66666667e-02, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-02, 3.00000000e+00, 2.29774000e-02]]),
            'r': np.array([
                 [5.00000000e-02, 3.00000000e+00, 2.00900000e-04],
                 [5.00000000e-02, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-02, 3.00000000e+00, 2.29774000e-02],
                 [1.66666667e-02, 3.00000000e+00, 2.00900000e-04],
                 [1.66666667e-02, 3.00000000e+00, 4.81823047e-03],
                 [1.66666667e-02, 3.00000000e+00, 2.29774000e-02],
                 [1.69524275e-02, 3.00000000e+00, 2.00900000e-04],
                 [1.69524275e-02, 3.00000000e+00, 4.81823047e-03],
                 [1.69524275e-02, 3.00000000e+00, 2.29774000e-02],
                 [5.00000000e-02, 3.00000000e+00, 2.00900000e-04],
                 [1.66666667e-02, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-02, 3.00000000e+00, 2.29774000e-02],
                 [5.00000000e-01, 3.00000000e+00, 2.00900000e-04],
                 [5.00000000e-01, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-01, 3.00000000e+00, 2.29774000e-02],
                 [5.00000000e-01, 3.00000000e+00, 2.00900000e-04],
                 [5.00000000e-01, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-01, 3.00000000e+00, 2.29774000e-02],
                 [5.00000000e-01, 3.00000000e+00, 2.00900000e-04],
                 [5.00000000e-01, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-01, 3.00000000e+00, 2.29774000e-02],
                 [5.00000000e-01, 3.00000000e+00, 2.00900000e-04],
                 [5.00000000e-01, 3.00000000e+00, 4.81823047e-03],
                 [5.00000000e-01, 3.00000000e+00, 2.29774000e-02]])
            }

        # specific to pearson, bonferroni
        self.resample_k = 1
        self.fold = True
        self.fold_value = {
            'p': 10,
            'r': 1
            }

        self.forward = True
        self.alpha = {
            'p': 0.05,
            'r': 0.5
            }

        self.paired = True
        self.n_corr = 6
        self.pvalues = self.assign_statistics_truths['pearson']['pvalues']
        self.correlations = self.assign_statistics_truths['pearson']['correlations']
        self.threshold = {
            'p': self.threshold_results['p'][5][0],
            'r': self.threshold_results['p'][5][0]
            }

        self.initial_corr = {
            'p': [(1, 0), (2, 1)],
            'r': [(1, 0), (2, 0), (2, 1)]
            }
        self.all_pairs = [(1, 0), (2, 0), (2, 1)]

        # specific to pearson, paired var1 = 1 var2 = 2 with bonferonni
        self.var1, self.var2 = 1, 2
        self.sign = np.sign(self.correlations[self.var1][self.var2])

        self.infln_metrics = ['cutie_1pc', 'cookd', 'dffits', 'dsr']
        self.infln_mapping = {
            'cutie_1pc': statistics.resample1_cutie_pc,
            'cookd': statistics.cookd,
            'dffits': statistics.dffits,
            'dsr': statistics.dsr
        }
        self.infln_results = {
            'p': {'cutie_1pc': set(),
                  'cookd': {(1, 0), (2, 1)},
                  'dffits': {(1, 0), (2, 1)},
                  'dsr': {(1, 0), (2, 1)}
                  },
            'r': {'cutie_1pc': set(),
                  'cookd': {(2, 0), (1, 0), (2, 1)},
                  'dffits': {(2, 0), (1, 0), (2, 1)},
                  'dsr': {(2, 0), (1, 0), (2, 1)}}
            }

        self.complete_pointwise_results = {
            'p':  ({'cutie_1pc': set(), 'cookd': {(1, 0), (2, 1)},
                    'dffits': {(1, 0), (2, 1)}, 'dsr': {(1, 0), (2, 1)}},
                    [['cutie_1pc'], ['cookd'], ['dffits'], ['dsr'],
                     ['cutie_1pc', 'cookd'], ['cutie_1pc', 'dffits'],
                     ['cutie_1pc', 'dsr'], ['cookd', 'dffits'], ['cookd', 'dsr'],
                     ['dffits', 'dsr'], ['cutie_1pc', 'cookd', 'dffits'],
                     ['cutie_1pc', 'cookd', 'dsr'], ['cutie_1pc', 'dffits', 'dsr'],
                     ['cookd', 'dffits', 'dsr'],
                     ['cutie_1pc', 'cookd', 'dffits', 'dsr']],
                     {"['cutie_1pc']": set(), "['cookd']": set(),
                      "['dffits']": set(), "['dsr']": set(),
                      "['cutie_1pc', 'cookd']": set(),
                      "['cutie_1pc', 'dffits']": set(),
                      "['cutie_1pc', 'dsr']": set(),
                      "['cookd', 'dffits']": set(), "['cookd', 'dsr']": set(),
                      "['dffits', 'dsr']": set(),
                      "['cutie_1pc', 'cookd', 'dffits']": set(),
                      "['cutie_1pc', 'cookd', 'dsr']": set(),
                      "['cutie_1pc', 'dffits', 'dsr']": set(),
                      "['cookd', 'dffits', 'dsr']": {(1, 0), (2, 1)},
                      "['cutie_1pc', 'cookd', 'dffits', 'dsr']": set()}),
            'r':  ({'cutie_1pc': set(), 'cookd': {(2, 0), (1, 0), (2, 1)},
                    'dffits': {(2, 0), (1, 0), (2, 1)},
                    'dsr': {(2, 0), (1, 0), (2, 1)}},
                    [['cutie_1pc'], ['cookd'], ['dffits'], ['dsr'],
                     ['cutie_1pc', 'cookd'], ['cutie_1pc', 'dffits'],
                     ['cutie_1pc', 'dsr'], ['cookd', 'dffits'], ['cookd', 'dsr'],
                     ['dffits', 'dsr'], ['cutie_1pc', 'cookd', 'dffits'],
                     ['cutie_1pc', 'cookd', 'dsr'], ['cutie_1pc', 'dffits', 'dsr'],
                     ['cookd', 'dffits', 'dsr'], ['cutie_1pc', 'cookd', 'dffits', 'dsr']],
                     {"['cutie_1pc']": set(), "['cookd']": set(),
                      "['dffits']": set(), "['dsr']": set(),
                      "['cutie_1pc', 'cookd']": set(),
                      "['cutie_1pc', 'dffits']": set(),
                      "['cutie_1pc', 'dsr']": set(), "['cookd', 'dffits']": set(),
                      "['cookd', 'dsr']": set(), "['dffits', 'dsr']": set(),
                      "['cutie_1pc', 'cookd', 'dffits']": set(),
                      "['cutie_1pc', 'cookd', 'dsr']": set(),
                      "['cutie_1pc', 'dffits', 'dsr']": set(),
                      "['cookd', 'dffits', 'dsr']": {(2, 0), (1, 0), (2, 1)},
                      "['cutie_1pc', 'cookd', 'dffits', 'dsr']": set()})
            }


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
            'p': {
                '1': (np.array([2., 2., 2., 2., 2.]),
                      np.array([0., 0., 0., 0., 0.]),
                      np.array([0.00319451, 0.00284677, 0.0030147 , 0.        , 0.05327074]),
                      np.array([0.99680549, 0.99715323, 0.9969853 , 1.        , 0.94672926])),
                '2': (np.array([8.00000e+00, 8.00000e+00, 8.00000e+00, 8.00000e+00, 8.00000e+00]),
                      np.array([0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00]),
                      np.array([3.33333e-01, 1.21038e-01, 2.12296e-01, 1.64309e-08, 3.33333e-01]),
                      np.array([8.66025e-01, 9.81981e-01, 9.44911e-01, 1.00000e+00, 8.66025e-01]))},
            'r': {
                '1': (np.array([2., 2., 2., 2., 2.]),
                      np.array([0., 0., 0., 0., 0.]),
                      np.array([0.00319451, 0.00284677, 0.0030147 , 0.        , 0.05327074]),
                      np.array([0.99680549, 0.99715323, 0.9969853 , 1.        , 0.94672926])),
                '2': (np.array([8.00000e+00, 8.00000e+00, 8.00000e+00, 8.00000e+00, 8.00000e+00]),
                      np.array([0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00]),
                      np.array([3.33333e-01, 1.21038e-01, 2.12296e-01, 1.64309e-08, 3.33333e-01]),
                      np.array([8.66025e-01, 9.81981e-01, 9.44911e-01, 1.00000e+00, 8.66025e-01]))}
            }

        self.forward_stats = ['pearson', 'spearman', 'kendall']
        self.reverse_stats = ['rpearson', 'rspearman', 'rkendall']

        self.evaluate_correlation_k_results = {
            'p': {
                '0': (np.array([2., 2., 2., 2., 2.]),
                      np.array([0., 0., 0., 0., 0.]),
                      0.05327073759374248, 0.9467292624062575),
                '1': (np.array([8., 8., 8., 8., 8.]),
                      np.array([0., 0., 0., 0., 0.]),
                      0.3333333333333333, 0.8660254037844387)},
            'r': {
                '0': (np.array([2., 2., 2., 2., 2.]),
                      np.array([0., 0., 0., 0., 0.]),
                      0.05327073759374248, 0.9467292624062575),
                '1': (np.array([8., 8., 8., 8., 8.]),
                      np.array([0., 0., 0., 0., 0.]),
                      0.3333333333333333, 0.8660254037844387)}}


        self.update_cutiek_true_corr_results = {
            'p': {
                '1': (defaultdict(list),
                  defaultdict(list),
                  {'1': [(1, 0)]},
                    {'1': np.array([[1.        , 1.        , 1.        ],
                                    [0.32580014, 1.        , 1.        ],
                                    [1.        , 0.05327074, 1.        ]])},
                    {'1': np.array([[ 0.        ,  0.        ,  0.        ],
                                    [-0.67419986,  0.        ,  0.        ],
                                    [ 0.        ,  0.94672926,  0.        ]])},
                  {'1': np.array([0., 0., 0., 0., 2.])},
                  {'1': np.array([0., 1., 1.])},
                  {'1': np.array([1., 1., 0.])},
                  {'1': {'(0, 1)': np.array([0., 0., 0., 0., 1.]),
                         '(1, 0)': np.array([0., 0., 0., 0., 1.]),
                         '(1, 2)': np.array([0., 0., 0., 0., 1.]),
                         '(2, 1)': np.array([0., 0., 0., 0., 1.])}},
                  {'1': {'(0, 1)': np.array([0., 0., 0., 0., 2.]),
                         '(1, 0)': np.array([0., 0., 0., 0., 2.]),
                         '(1, 2)': np.array([0., 0., 0., 0., 0.]),
                         '(2, 1)': np.array([0., 0., 0., 0., 0.])}})},
            'r': {
                '1': (defaultdict(list),
                  defaultdict(list),
                  {'1': []},
                    {'1': np.array([[1.        , 1.        , 1.        ],
                                    [0.32580014, 1.        , 1.        ],
                                    [0.24566349, 0.05327074, 1.        ]])},
                    {'1': np.array([[ 0.        ,  0.        ,  0.        ],
                                    [-0.67419986,  0.        ,  0.        ],
                                    [-0.75433651,  0.94672926,  0.        ]])},
                  {'1': np.array([0., 0., 0., 0., 0.])},
                  {'1': np.array([0., 0., 0.])},
                  {'1': np.array([0., 0., 0.])},
                  {'1': {'(0, 1)': np.array([0., 0., 0., 0., 0.]),
                         '(1, 0)': np.array([0., 0., 0., 0., 0.]),
                         '(1, 2)': np.array([0., 0., 0., 0., 0.]),
                         '(2, 1)': np.array([0., 0., 0., 0., 0.]),
                         '(2, 0)': np.array([0., 0., 0., 0., 0.])}},
                  {'1': {'(0, 1)': np.array([0., 0., 0., 0., 2.]),
                         '(1, 0)': np.array([0., 0., 0., 0., 2.]),
                         '(1, 2)': np.array([0., 0., 0., 0., 0.]),
                         '(2, 1)': np.array([0., 0., 0., 0., 0.]),
                         '(2, 0)': np.array([0., 0., 0., 0., 2.])}})}
            }

        # setup pointwise results with intermediate files
        self.test_dir = os.path.abspath(os.path.dirname(__file__))
        self.work_dir = os.path.join(self.test_dir, 'test_data/')

        # setup pointwise results with intermediate files for negnan
        tuplesP = list(itertools.permutations(list(range(len(self.samp_var1[0,:]))), 2))
        tuplesC = list(itertools.combinations_with_replacement(list(range(len(self.samp_var2[0,:]))), 2))
        self.tuples  = sorted(list(set(tuplesP) | set(tuplesC)))


    def test_compute_pc(self):
        assert_almost_equal((np.nan, np.nan), statistics.compute_pc(self.undef_corr[0],
                                                         self.undef_corr[1]))
        assert_almost_equal((0, 1), statistics.compute_pc(self.perfect_corr[0],
                                                         self.perfect_corr[1]))
        assert_almost_equal((0.333, 0.866), statistics.compute_pc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)

    def test_compute_sc(self):
        assert_almost_equal((np.nan, np.nan), statistics.compute_sc(self.undef_corr[0],
                                                         self.undef_corr[1]))
        assert_almost_equal((0, 1), statistics.compute_sc(self.perfect_corr[0],
                                                         self.perfect_corr[1]))
        assert_almost_equal((0.666,0.5), statistics.compute_sc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)

    def test_compute_kc(self):
        assert_almost_equal((np.nan, np.nan), statistics.compute_kc(self.undef_corr[0],
                                                         self.undef_corr[1]))
        assert_almost_equal((0.333, 1), statistics.compute_kc(
            self.perfect_corr[0], self.perfect_corr[1]), decimal=3)
        assert_almost_equal((1, 0.333), statistics.compute_kc(
            self.empirical_corr[0], self.empirical_corr[1]), decimal=3)


    def test_initial_stats(self):
        for stat in self.functions:
            assert_almost_equal(self.true_paired_sampvar1_arrays[stat],
                statistics.initial_stats(self.samp_var1, self.samp_var1,
                    self.mapf[stat], paired=True), decimal=7)

    def test_assign_statistics(self):
        keys = ['pvalues','correlations','r2vals']
        # tests pearson, spearman and kendall
        for stat in self.correlation_types:
            stats_vals = statistics.assign_statistics(self.samp_var1,
                self.samp_var1, stat, self.pearson_stats, self.spearman_stats,
                self.kendall_stats, paired=True)

            for k in range(len(keys)):
                assert_almost_equal(self.assign_statistics_truths[stat][keys[k]],
                                    stats_vals[k], decimal=7)


        # assertRaise for valuerror, 'kpc' not a valid stat string
        self.assertRaises(ValueError, statistics.assign_statistics,
            self.samp_var1, self.samp_var2, 'kpc',
            self.pearson_stats, self.spearman_stats, self.kendall_stats,
            paired=True)

    def test_set_threshold(self):
        # test the different allowed multiple corrections adjustments
        mc_types = ['nomc', 'bonferroni', 'fwer', 'fdr']
        results = []
        for p in ['p', 'r']:
            for mc in mc_types:
                for stat in self.correlation_types:
                    results.append(statistics.set_threshold(
                        self.assign_statistics_truths[stat]['pvalues'], p,
                        self.alpha[p], mc, self.paired))

            assert_almost_equal(self.threshold_results[p], np.array(results), decimal=7)

    def test_get_initial_corr(self):
        # modified from prev branch to incorporate different statistic parameter
        for p in ['p', 'r']:
            assert (self.initial_corr[p], self.all_pairs) == statistics.get_initial_corr(
                self.n_var1, self.n_var2, self.pvalues, self.correlations,
                self.threshold[p], p, self.paired)

    # no unit test written for return_influence() because the return vars
    # are objects of the sm.OLS class

    def test_calculate_FP_sets(self):
        # True for self.fold
        # modified from prev branch to incorporate different statistic parameter
        for p in ['p', 'r']:
            assert self.infln_results[p] == statistics.calculate_FP_sets(self.initial_corr[p],
            self.samp_var1, self.samp_var2, self.infln_metrics, self.infln_mapping,
            self.threshold[p], True, self.fold_value[p], p)

    def test_pointwise_metrics(self):
        # generate results and output intermediate file
        pointwise_results = {}
        for t in self.tuples:
            t1, t2, = t
            pointwise_results[str(t)] = {}

            for p in ['p', 'r']:
                pointwise_results[str(t)][p] = {}

                for f in self.infln_metrics:#self.infln_mapping:
                    x_old = self.samp_var1[:, t1]
                    y_old = self.samp_var2[:, t2]

                    var1_values, var2_values = utils.remove_nans(x_old, y_old)
                    influence = statistics.return_influence(var1_values, var2_values)

                    arr_0, arr_1, arr_2, arr_3 = self.infln_mapping[f](var1_index=t1,
                        var2_index=t2, samp_var1=self.samp_var1, samp_var2=self.samp_var2,
                        influence=influence, threshold=self.threshold[p], fold=self.fold,
                        fold_value=self.fold_value[p], param=p)

                    # save results to compressed object and later text file
                    fp = self.work_dir + '_'.join([str(t), p, f, '.npz'])
                    np.savez(fp, arr_0, arr_1, arr_2, arr_3)
                    results = []
                    for key, value in np.load(fp).items():
                        results.append(value)
                        np.savetxt(self.work_dir + '_'.join([str(t), p, f, key + '.txt']), value)

                    pointwise_results[str(t)][p][f] = results

        # test cutie, cookd, dffits, dsr
        # with inputs mixed with nan and neg values as defined in setUp
        for t in self.tuples:
            t1, t2 = t
            var1_values = self.samp_var1[:, t1]
            var2_values = self.samp_var2[:, t2]
            influence = statistics.return_influence(var1_values, var2_values)

            for p in ['p', 'r']:
                for f in self.infln_mapping:
                    results = self.infln_mapping[f](var1_index=t1,
                        var2_index=t2, samp_var1=self.samp_var1, samp_var2=self.samp_var2,
                        influence=influence,
                        threshold=self.threshold[p], fold=self.fold,
                        fold_value=self.fold_value[p], param=p)
                    # comparison to 7 decimal places is the default value
                    assert_almost_equal(pointwise_results[str(t)][p][f], results)

    def test_pointwise_comparison(self):
        for p in ['p', 'r']:
            assert self.complete_pointwise_results[p] == statistics.pointwise_comparison(
                self.infln_metrics, self.infln_mapping, self.samp_var1,
                self.samp_var2, self.initial_corr[p], self.threshold[p],
                self.fold_value[p], self.fold, p)


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
        for p in ['p', 'r']:
            for k in [1,2]:
                assert_almost_equal(self.resamplek_results[p][str(k)],
                    statistics.resamplek_cutie(self.var1, self.var2, self.n_samp,
                        self.samp_var1, self.samp_var1,self.pvalues, self.correlations,
                        self.threshold[p], k, self.sign, self.forward, 'pearson', self.fold,
                        self.fold_value[p], p),decimal=5)

    def test_evaluation_correlation_k(self):
        # 0, 1 as opposed to 1, 2 because you add 1 inside evaluate_correlation_k()
        for p in ['p','r']:
            for k in [0,1]:
                results = statistics.evaluate_correlation_k(
                    self.var1, self.var2, self.n_samp, self.samp_var1,
                    self.samp_var1, self.pvalues, self.correlations, self.threshold[p],
                    'pearson', k, self.sign, self.fold, self.fold_value[p],
                    self.forward, p)
                for r in range(len(results)):
                    assert_almost_equal(results[r],
                        self.evaluate_correlation_k_results[p][str(k)][r])

    def test_update_cutiek_true_corr(self):
        statistic = 'pearson'
        for p in ['p','r']:
            for k in [1]:
                results = statistics.update_cutiek_true_corr(self.initial_corr[p],
                    self.samp_var1, self.samp_var1, self.pvalues, self.correlations,
                    self.threshold[p], 'pearson', self.forward_stats,
                    self.reverse_stats, k, self.fold, self.fold_value[p], p)

                for r in range(2,len(results)):
                    # if not dictionary
                    if not isinstance(results[r][str(k)], collections.Mapping):
                        assert_almost_equal(results[r][str(k)],
                            self.update_cutiek_true_corr_results[p][str(k)][r][str(k)])
                    else:
                        for key in results[r][str(k)].keys():
                            assert_almost_equal(results[r][str(k)][key],
                                self.update_cutiek_true_corr_results[p][str(k)][r][str(k)][key])

if __name__ == '__main__':
    unittest.main()
