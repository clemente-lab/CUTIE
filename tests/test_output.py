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

        with open(devnull, 'w') as dn:
            call('mkdir ' + OutputTest.work_dir, stderr=dn, shell=True)

    @classmethod
    def tearDownClass(OutputTest):
        with open(devnull, 'w') as dn:
            call('rm -r ' + OutputTest.work_dir + '*', stderr=dn, shell=True)

    def setUp(self):
        """ Call before each test to ensure a clean working environment """
        with open(devnull, 'w') as dn:
            call('rm -r ' + self.work_dir + '*', stderr=dn, shell=True)

    def test_print_summary_df(self):
        data_processing = os.path.join(self.work_dir, 'data_processing/')
        with open(devnull, 'w') as dn:
            call('mkdir ' + data_processing, stderr=dn, shell=True)

        df1 = output.print_summary_df(3, 3, ['pvalues', 'indicators', 'FP_rev_indicators', 'TP_rev_indicators'],
                                      [self.simple_pval_matrix, self.simple_pval_matrix, self.simple_pval_matrix, self.simple_pval_matrix], self.work_dir,
                                      1, 3, paired=True)
        np.testing.assert_array_equal(self.test_summary_df, df1.values)

        df2 = output.print_summary_df(0, 0, self.empty_array1, self.empty_array1,
                                      self.work_dir, 1, 0, paired=True)
        matrix2 = df2.values
        matrix2.shape = (0,)
        np.testing.assert_almost_equal(self.empty_array1, matrix2)

if __name__ == "__main__":
    main()
