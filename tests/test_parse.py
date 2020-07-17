#!/usr/bin/python

import unittest
import os
import numpy as np
from cutie import parse
from numpy.testing import assert_almost_equal, assert_equal


class TestStatistics(unittest.TestCase):
    # setUp name is specific to package
    def setUp(self):

        self.minep_fp = os.path.dirname(os.path.realpath(__file__)) + \
            '/n=50,alpha=0.6.csv'
        self.f2type = 'map'
        self.f1type = 'otu'
        self.skip1 = 1
        self.skip2 = 0
        self.startcol1 = 5
        self.endcol1 = 10
        self.startcol2 = 17
        self.endcol2 = 19
        self.delimiter1 = '\t'
        self.delimiter2 = '\t'
        self.samp_var1_fp = os.path.dirname(os.path.realpath(__file__)) + \
            '/otu_table_MultiO_merged___L6.txt'
        self.samp_var2_fp = os.path.dirname(os.path.realpath(__file__)) + \
        '/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt'
        with open(self.minep_fp, 'r') as f:
            self.mine_bins, self.pvalue_bins = parse.parse_minep(f, ',', 13)

        self.process_df_results = {
            '1': (np.array([[0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.00296736, 0.        , 0.        ],
                       [0.        , 0.00702201, 0.00016587, 0.        , 0.        ],
                       [0.        , 0.00013331, 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.0001419 , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.00049568, 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.00236432, 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        , 0.        ]]),
                np.array([8.44399286e-05, 2.73249893e-04, 1.11901179e-04, 5.06800000e-06,
                        0.00000000e+00]),
                np.array([1.92512742e-07, 1.69576445e-06, 3.02932898e-07, 6.93484848e-10,
                        0.00000000e+00])),
            '2': (np.array([[  60088.1214 ,  811001.3592 ],
                           [ 120075.1344 , 1931721.553  ],
                           [  19712.98   ,  156929.4015 ],
                           [  17490.178  , 1268706.647  ],
                           [  73103.15772,  193792.1451 ],
                           [  91490.47715,   45931.19482],
                           [  74320.1465 , 1801787.517  ],
                           [ 223226.1232 ,  115462.7898 ],
                           [  65152.37963,   44999.19709],
                           [  18206.5968 ,   93603.8229 ],
                           [  55141.21631,   41917.3047 ],
                           [  17355.9223 ,  233084.0972 ],
                           [  58504.80283,   42146.92313],
                           [ 110596.5645 , 1117430.183  ],
                           [ 272839.3202 , 2580014.534  ],
                           [ 322618.1188 , 1786407.564  ],
                           [ 114864.323  , 1019225.174  ],
                           [ 215248.4388 , 4010056.413  ],
                           [  57435.66899,   31882.41563],
                           [  97961.5624 , 1257122.001  ],
                           [  32662.6013 ,  235601.3443 ],
                           [  53034.4714 , 1266819.658  ],
                           [ 186826.4635 , 2687662.923  ],
                           [ 309002.928  , 9278219.496  ],
                           [ 160898.0428 ,  153451.6236 ],
                           [ 289463.802  , 5687572.542  ],
                           [ 193512.0719 , 2382798.92   ],
                           [ 142062.84   , 2065672.617  ]]),
                    np.array([ 123317.65906536, 1512179.33432036]),
                    np.array([8.50279265e+09, 4.01598194e+12]))}

    def test_parse_input(self):
        # the output of results is as follows
        # samp_ids, var_names, df, n_var, n_samp
        results = parse.parse_input(self.f2type, self.samp_var2_fp, self.startcol2,
            self.endcol2, self.delimiter2, self.skip2)
        # testing samp_ids match
        for i in range(len(results[0])):
            assert results[0][i] == [
                '100716FG.C.1.RL', '100804MB.C.1.RL', '100907LG.C.1.RL',
                '101007PC.C.1.RL', '101018WG.N.1.RL', '101019AB.N.1.RL',
                '101026RM.C.1.RL', '101109JD.N.1.RL', '101206DM.N.1.RL',
                '110222MG.C.1.RL', '110228CJ.N.1.RL', '110308DK.C.1.RL',
                '110314CS.N.1.RL', '110330DS.C.1.RL', '110406MB.C.1.RL',
                '110412ET.C.1.RL', '110418ML.C.1.RL', '110420JR.C.1.RL',
                '110502BC.N.1.RL', '110523CB.C.1.RL', '110601OG.C.1.RL',
                '110720BB.C.1.RL', '110727MK.C.1.RL', '110801EH.C.1.RL',
                '110808JB.N.1.RL', '110921AR.C.1.RL', '111003JG.C.1.RL',
                '111115WK.C.1.RL'][i]
        # testing var_names match
        for i in range(len(results[1])):
            assert results[1][i] == ['glutamic_acid', 'glycine'][i]

        # testing n_var and n_samp match
        assert results[3] == 2
        assert results[4] == 28

        results = parse.parse_input(self.f1type, self.samp_var1_fp, self.startcol1,
            self.endcol1, self.delimiter1, self.skip1)
        # testing samp_ids match
        for i in range(len(results[0])):
            assert results[0][i] == [
                '101019AB.N.1.RL', '110228CJ.N.1.RL', '110314CS.N.1.RL',
                '110502BC.N.1.RL', '110808JB.N.1.RL', '101018WG.N.1.RL',
                '101109JD.N.1.RL', '101206DM.N.1.RL', '100907LG.C.1.RL',
                '110308DK.C.1.RL', '110412ET.C.1.RL', '110418ML.C.1.RL',
                '110601OG.C.1.RL', '110720BB.C.1.RL', '110727MK.C.1.RL',
                '110801EH.C.1.RL', '110921AR.C.1.RL', '111003JG.C.1.RL',
                '111115WK.C.1.RL', '100804MB.C.1.RL', '100716FG.C.1.RL',
                '101007PC.C.1.RL', '101026RM.C.1.RL', '110222MG.C.1.RL',
                '110330DS.C.1.RL', '110406MB.C.1.RL', '110420JR.C.1.RL',
                '110523CB.C.1.RL'][i]

        # testing var_names match
        for i in range(len(results[1])):
            assert results[1][i] == [
                'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__',
                'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium',
                'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobrevibacter',
                'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanothermobacter',
                'k__Archaea;p__Euryarchaeota;c__Methanomicrobia;o__Methanocellales;f__Methanocellaceae;g__Methanocella'][i]

        # testing n_var and n_samp match
        assert results[3] == 5
        assert results[4] == 28

    def test_process_df(self):
        # test processing of dataframes
        samp_ids, var_names, df, n_var, n_samp = parse.parse_input(self.f1type,
            self.samp_var1_fp, self.startcol1, self.endcol1, self.delimiter1, self.skip1)

        results = parse.process_df(df, samp_ids)
        for r in range(len(results)):
            assert_almost_equal(self.process_df_results['1'][r], results[r])

        samp_ids, var_names, df, n_var, n_samp = parse.parse_input(self.f2type,
            self.samp_var2_fp, self.startcol2, self.endcol2, self.delimiter2, self.skip2)

        results = parse.process_df(df, samp_ids)
        for r in range(len(results)):
            assert_almost_equal(self.process_df_results['2'][r], results[r],decimal=-5)

    def test_parse_minep(self):
        # test for equality of first 5 entries of mine p file path
        mine_results = (np.array([0.8058,  0.78131, 0.76758, 0.75606, 0.7494 ]),
            np.array([2.56e-07, 5.12e-07, 7.68e-07, 1.024e-06, 1.281e-06]))

        with open(self.minep_fp, 'r') as f:
            results = parse.parse_minep(f, delimiter=',', pskip=13)

        for r in range(len(results)):
            assert_almost_equal(results[r][0:5], mine_results[r])


if __name__ == '__main__':
    unittest.main()
