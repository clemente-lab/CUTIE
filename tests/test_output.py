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
        OutputTest.headers1 = ['head1', 'head2', 'head3', 'head4', 'head5']
        OutputTest.headers2 = ['var1_index', 'var2_index', 'avg_var1',
                                        'avg_var2', 'var_var1', 'var_var2',
                                        'pvalues']
        OutputTest.headers3 = ['var1_index', 'var2_index', 'avg_var1',
                                        'avg_var2', 'var_var1', 'var_var2']
        OutputTest.base_regions1 = ['cutie_1pc', 'cookd', 'dffits', 'dsr']
        OutputTest.regions_set1 = {'cutie_1pc': [(0, 1), (1, 0), (2, 1), (1, 2)],
                                   'cookd': [(0, 1), (1, 0), (2, 1), (1, 2)],
                                   'dffits': [(0, 1), (1, 0), (2, 1), (1, 2)],
                                   'dsr': [(1, 0), (2, 1), (1, 2)]}
        OutputTest.empty_array1 = np.array([])
        OutputTest.empty_array2 = np.array([[]])
        OutputTest.simple_array = np.array([1, 2, 3])
        OutputTest.simple_pval_matrix = np.array([[1, 1, 1], [1, 1, 1],
                                                  [0, 0, 0]])
        OutputTest.even_array1 = np.array([[1, 2, 3, 4, 5],
                                           [5, 4, 3, 2, 1],
                                           [6, 7, 8, 9, 0],
                                           [0, 9, 8, 7, 6]])
        OutputTest.uneven_array = np.array([[1, 2, 3, 4, 5],
                                            [1, 2],
                                            [3, 4, 5, 6, 7, 8, 9, 0]])
        OutputTest.test_Rmatrix = np.array([[0, 1, 1, 2, 1, 2, 1],
                                            [0, 2, 1, 3, 1, 3, 1],
                                            [1, 0, 2, 1, 2, 1, 1],
                                            [1, 2, 2, 3, 2, 3, 1],
                                            [2, 0, 3, 1, 3, 1, 0],
                                            [2, 1, 3, 2, 3, 2, 0]])
        OutputTest.test_tuple_list1 = set([(1, 2), (3, 1), (4, 2), (2, 1),
                                          (1, 3), (2, 4)])
        OutputTest.test_tuple_list2 = set([(3, 5), (2, 4), (1, 8), (8, 1),
                                          (4, 2), (5, 3)])
        OutputTest.test_tuple_list3 = set([(1, 2), (3, 5), (2, 4), (2, 1),
                                           (5, 3), (4, 2)])
        OutputTest.test_tuple_dict1 = {'1':OutputTest.test_tuple_list1,
                                       '2': OutputTest.test_tuple_list2}

        OutputTest.test_dir = os.path.abspath(os.path.dirname(__file__))
        OutputTest.work_dir = os.path.join(OutputTest.test_dir, 'temp/')
        OutputTest.data_dir = os.path.join(OutputTest.test_dir, 'data/')
        OutputTest.empty_file = os.path.join(OutputTest.data_dir, 'empty.txt')
        OutputTest.test_matrix1 = os.path.join(OutputTest.data_dir,
                                               'test_matrix1.txt')
        OutputTest.test_matrix2 = os.path.join(OutputTest.data_dir,
                                               'test_matrix2.txt')
        OutputTest.test_matrix3 = os.path.join(OutputTest.data_dir,
                                               'test_matrix3.txt')
        OutputTest.false_corr1 = os.path.join(OutputTest.data_dir,
                                              'false_corr1.txt')
        OutputTest.false_corr2 = os.path.join(OutputTest.data_dir,
                                              'false_corr2.txt')
        OutputTest.true_corr1 = os.path.join(OutputTest.data_dir,
                                             'true_corr1.txt')
        OutputTest.true_corr2 = os.path.join(OutputTest.data_dir,
                                             'true_corr2.txt')
        OutputTest.all_pairs1 = os.path.join(OutputTest.data_dir,
                                             'all_pairs.txt')
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

    def test_print_Rmatrix(self):
        data_processing = os.path.join(self.work_dir, 'data_processing/')
        with open(devnull, 'w') as dn:
            call('mkdir ' + data_processing, stderr=dn, shell=True)
        matrix1, test_headers1 = output.print_Rmatrix(self.simple_array,
                                                      self.simple_array,
                                                      self.simple_array,
                                                      self.simple_array, 3, 3,
                                                      ['pvalues'],
                                                      [self.simple_pval_matrix],
                                                      self.work_dir, 1, 'test',
                                                      6, paired=True)
        np.testing.assert_array_equal(self.test_Rmatrix, matrix1)
        np.testing.assert_array_equal(self.headers2, test_headers1)
        matrix2, test_headers2 = output.print_Rmatrix(self.empty_array1,
                                                      self.empty_array1,
                                                      self.empty_array1,
                                                      self.empty_array1, 0, 0,
                                                      self.empty_array1,
                                                      self.empty_array1,
                                                      self.work_dir, 1,
                                                      'test', 0)
        matrix2.shape = (0,)
        np.testing.assert_almost_equal(self.empty_array1, matrix2)

    def test_print_true_false_corr(self):
        data_processing = os.path.join(self.work_dir, 'data_processing/')
        with open(devnull, 'w') as dn:
            call('mkdir ' + data_processing, stderr=dn, shell=True)
        test_file1 = os.path.join(data_processing, 'testlog1_falsesig.txt')
        test_file2 = os.path.join(data_processing, 'testlog2_falsesig.txt')
        test_file3 = os.path.join(data_processing, 'testlog1_truesig.txt')
        test_file4 = os.path.join(data_processing, 'testlog2_truesig.txt')
        output.print_true_false_corr(self.test_tuple_list3,
                                     self.test_tuple_dict1,
                                     self.work_dir, 'test', 2, 'log')

        self.assertTrue(filecmp.cmp(test_file1, self.false_corr1, shallow=True))
        self.assertTrue(filecmp.cmp(test_file2, self.false_corr2, shallow=True))
        self.assertTrue(filecmp.cmp(test_file3, self.true_corr1, shallow=True))
        self.assertTrue(filecmp.cmp(test_file4, self.true_corr2, shallow=True))

    def test_generate_pair_matrix(self):
        data_processing = os.path.join(self.work_dir, 'data_processing/')
        with open(devnull, 'w') as dn:
            call('mkdir ' + data_processing, stderr=dn, shell=True)
        test_file1 = os.path.join(data_processing, 'all_pairs.txt')
        output.generate_pair_matrix(self.base_regions1, self.regions_set1, 3, 3,
                                    self.work_dir)

if __name__ == "__main__":
    main()
