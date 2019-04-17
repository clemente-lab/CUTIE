import filecmp
import os.path
from os import devnull
from subprocess import call
from unittest import TestCase, main


import pandas as pd
import numpy as np

from cutie import output


testing_dir = 'temp/'

class OutputTest(TestCase):

    @classmethod
    def setUpClass(OutputTests):
        OutputTest.headers = ['head1', 'head2', 'head3', 'head4', 'head5']
        OutputTest.empty_array1 = []
        OutputTest.empty_array2 = [[]]
        OutputTest.even_array1 = [[1, 2, 3, 4, 5],
                                  [5, 4, 3, 2, 1],
                                  [6, 7, 8, 9, 0],
                                  [0, 9, 8, 7, 6]]
        OutputTest.uneven_array = [[1, 2, 3, 4, 5],
                                   [1, 2],
                                   [3, 4, 5, 6, 7, 8, 9, 0]]

        OutputTest.test_dir = os.path.abspath(os.path.dirname(__file__))
        OutputTest.data_dir = os.path.join(OutputTest.test_dir, '\\data\\')
        OutputTest.empty_file = os.path.join(OutputTest.data_dir, 'empty.txt')
        OutputTest.test_matrix1 = os.path.join(OutputTest.data_dir,
                                               'test_matrix1.txt')
        OutputTest.test_matrix2 = os.path.join(OutputTest.data_dir,
                                               'test_matrix2.txt')
        OutputTest.test_matrix3 = os.path.join(OutputTest.data_dir,
                                               'test_matrix3.txt')

    @classmethod
    def tearDownClass(PlotTests):
        with open(devnull, 'w') as dn:
            call('rm -r ' + testing_dir + '*', stderr=dn, shell=True)

    def setUp(self):
        """ Call before each test to ensure a clean working environment """
        with open(devnull, 'w') as dn:
            call('rm -r ' + testing_dir + '*', stderr=dn, shell=True)

    def test_print_matrix(self):
        empty1 = testing_dir + 'empty1.txt'
        empty2 = testing_dir + 'empty2.txt'
        even1 = testing_dir + 'even1.txt'
        even2 = testing_dir + 'even2.txt'
        uneven1 = testing_dir + 'uneven1.txt'

        output.print_matrix(self.empty_array1, empty1, self.empty_array1, '\t')
        output.print_matrix(self.empty_array2, empty2, self.empty_array1, '\t')
        output.print_matrix(self.even_array1, even1, self.headers, '\t')
        output.print_matrix(self.even_array1, even2, self.empty_array1, ':')
        output.print_matrix(self.uneven_array, uneven1, self.empty_array1, '\t')
        self.assertTrue(filecmp.cmp(open(empty1, 'r'),
                                    open(self.empty_file, 'r'),
                                    shallow=False))
        self.assertTrue(filecmp.cmp(open(empty2, 'r'),
                                    open(self.empty_file, 'r'),
                                    shallow=False))
        self.assertTrue(filecmp.cmp(open(even1, 'r'),
                                    open(self.test_matrix1, 'r'),
                                    shallow=False))
        self.assertTrue(filecmp.cmp(open(even2, 'r'),
                                    open(self.test_matrix2, 'r'),
                                    shallow=False))
        self.assertTrue(filecmp.cmp(open(uneven1, 'r'),
                                    open(self.test_matrix3, 'r'),
                                    shallow=False))
