import filecmp
import os.path
from os import devnull
from subprocess import call
from unittest import TestCase, main

import numpy as np

from cutie import output


class OutputTest(TestCase):

    @classmethod
    def setUpClass(OutputTest):
        OutputTest.empty_array1 = np.array([])
        OutputTest.simple_array = np.array([1, 2, 3])
        OutputTest.simple_pval_matrix = np.array([[1, 1, 0],
                                                  [1, 1, 0],
                                                  [0, 0, 1]])

        OutputTest.test_summary_df = np.array([[1, 0, 1, 1, 1, 1],
                                               [2, 0, 0, 0, 0, 0],
                                               [2, 1, 0, 0, 0, 0]])


        OutputTest.test_dir = os.path.abspath(os.path.dirname(__file__))
        OutputTest.work_dir = os.path.join(OutputTest.test_dir, 'temp/')

        OutputTest.samp_ids =  ['101019AB.N.1.RL',
                                '110228CJ.N.1.RL',
                                '110314CS.N.1.RL',
                                '110502BC.N.1.RL',
                                '110808JB.N.1.RL',
                                '101018WG.N.1.RL',
                                '101109JD.N.1.RL',
                                '101206DM.N.1.RL',
                                '100907LG.C.1.RL',
                                '110308DK.C.1.RL',
                                '110412ET.C.1.RL',
                                '110418ML.C.1.RL',
                                '110601OG.C.1.RL',
                                '110720BB.C.1.RL',
                                '110727MK.C.1.RL',
                                '110801EH.C.1.RL',
                                '110921AR.C.1.RL',
                                '111003JG.C.1.RL',
                                '111115WK.C.1.RL',
                                '100804MB.C.1.RL',
                                '100716FG.C.1.RL',
                                '101007PC.C.1.RL',
                                '101026RM.C.1.RL',
                                '110222MG.C.1.RL',
                                '110330DS.C.1.RL',
                                '110406MB.C.1.RL',
                                '110420JR.C.1.RL',
                                '110523CB.C.1.RL']

        OutputTest.var1_names = [
            'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__',
            'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium',
            'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobrevibacter',
            'k__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanothermobacter',
            'k__Archaea;p__Euryarchaeota;c__Methanomicrobia;o__Methanocellales;f__Methanocellaceae;g__Methanocella'
            ]

        OutputTest.var2_names = ['glutamic_acid',
                                'glycine',
                                'alanine',
                                'succinic_acid',
                                'aspartic_acid',
                                'glutamine',
                                'serine',
                                'methionine',
                                'urea',
                                'sucrose',
                                'glycerol_alpha_phosphate',
                                'fructose',
                                'cysteine',
                                'beta_alanine',
                                'glycerol',
                                'ribose',
                                'fumaric_acid',
                                'leucine',
                                'proline',
                                'malic_acid',
                                'nicotinamide',
                                '4_hydroxybenzoate',
                                'citric_acid',
                                'mannose',
                                'glycolic_acid',
                                'thymine',
                                'benzoic_acid',
                                'valine',
                                'cellobiose',
                                'lactic_acid',
                                'cholesterol',
                                'threonine',
                                'ethanolamine'
                                ]
        OutputTest.samp_counter = {'1': np.array([0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])}
        OutputTest.var1_counter = {'1': np.array([0, 1, 2, 0, 0])}
        OutputTest.var2_counter = {'1': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])}

        with open(devnull, 'w') as dn:
            call('mkdir ' + OutputTest.work_dir, stderr=dn, shell=True)

    @classmethod
    def tearDownClass(OutputTest):
        with open(devnull, 'w') as dn:
            call('rm -r ' + OutputTest.work_dir, stderr=dn, shell=True)

    def setUp(self):
        """ Call before each test to ensure a clean working environment """
        with open(devnull, 'w') as dn:
            call('rm -r ' + self.work_dir + '*', stderr=dn, shell=True)

    def test_print_summary_df(self):
        data_processing = os.path.join(self.work_dir, 'data_processing/')
        with open(devnull, 'w') as dn:
            call('mkdir ' + data_processing, stderr=dn, shell=True)
        df1 = output.print_summary_df(3, 3, ['pvalues', 'indicators', 'FP_rev_indicators', 'TP_rev_indicators'],
                                      [self.simple_pval_matrix, self.simple_pval_matrix, self.simple_pval_matrix, self.simple_pval_matrix],
                                      self.work_dir, 1, 3, paired=True)
        np.testing.assert_array_equal(self.test_summary_df, df1.values)

        df2 = output.print_summary_df(0, 0, self.empty_array1, self.empty_array1,
                                      self.work_dir, 1, 0, paired=True)
        matrix2 = df2.values
        matrix2.shape = (0,)
        np.testing.assert_almost_equal(self.empty_array1, matrix2)

    def test_diag_plots(self):
        data_processing = os.path.join(self.work_dir, 'data_processing')
        with open(devnull, 'w') as dn:
            call('mkdir ' + data_processing, stderr=dn, shell=True)
        graphs = os.path.join(self.work_dir, 'graphs')
        with open(devnull, 'w') as dn:
            call('mkdir ' + graphs, stderr=dn, shell=True)

        output.diag_plots(self.samp_ids, self.var1_names, self.var2_names, self.samp_counter,
            self.var1_counter, self.var2_counter, 1, self.work_dir, False, 'kendall', ['pearson','spearman','kendall'])

        test_files = [
            'counter_var2_number_resample1.txt',
            'counter_var1_number_resample1.txt',
            'counter_samp_number_resample1.txt'
        ]

        for f in test_files:
            test_file = os.path.join(data_processing, f)
            self.assertTrue(filecmp.cmp(test_file, os.path.join(self.test_dir, 'test_data', f), shallow=True))



if __name__ == "__main__":
    main()
