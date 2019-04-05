#!/usr/bin/env python
from __future__ import division

from cutie import parse
from cutie import output

import numpy as np
from numpy.random import seed

from scipy.stats.distributions import gamma, lognorm, norm
from scipy.linalg import toeplitz

import correlations
from correlations import generators
from correlations.generators.ecological import *
from correlations.generators.copula import copula, generate_rho_matrix

import biom.table
from biom.parse import parse_biom_table

import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--working_dir', required=False,
              type=click.Path(exists=True),
              help='Directory to save simulation files')
@click.option('-s', '--seedn', required=False,
              type=int, default=0,
              help='Seed for random number generator')

def gen_cutie_copula(working_dir, seedn):
    """
    Generation of simulated datasets via copula method. See Weiss et al. 2016.
    ----------------------------------------------------------------------------
    INPUTS
    working_dir - String. File path in which to save simulated data. 
                  Should end in '/'
    """

    # set parameters for data
    n_otu = 500
    mu_mat = np.array([0]*n_otu)
    n1 = 50
    lognorm_methods = [[lognorm, 1, 0]]*n_otu
    gamma_methods = [[gamma, 1, 0, 100]]*n_otu
    norm_methods = [[norm, 0, 1]] * n_otu

    seed(seedn)

    # create covariance matrices
    # https://blogs.sas.com/content/iml/2015/09/23/large-spd-matrix.html
    dep_top = toeplitz(np.arange(1.0, -1.0, -2.0/n_otu))

    # create simulated data structures
    seed(seedn)
    copula_table1_n50_lognorm_3_0 = copula(n1, dep_top, mu_mat, 
        lognorm_methods)
    seed(seedn)
    copula_table2_n50_gamma_1_0_100 = copula(n1, dep_top, mu_mat, 
        gamma_methods)
    seed(seedn)
    copula_table3_n50_norm_0_1 = copula(n1, dep_top, mu_mat, norm_methods)


    tables = [copula_table1_n50_lognorm_3_0,
              copula_table2_n50_gamma_1_0_100,
              copula_table3_n50_norm_0_1]

    names = ['seed' + str(seedn) + '_copula_n50_lognorm_3_0.txt',
        'seed' + str(seedn) + '_copula_n50_gamma_1_0_100.txt', 
        'seed' + str(seedn) + '_copula_n50_norm_0_1.txt']

    def make_ids(data):
        ''' Helper function to create col and row names
        '''
        sids = ['s%i' % i for i in range(data.shape[1])]
        oids = ['o%i' % i for i in range(data.shape[0])]
        return sids, oids

    # print tables
    for table, name in zip(tables,names):
        sids, oids = make_ids(table)
        bt = biom.table.Table(table, oids, sids)
        output.print_matrix(dep_top, working_dir + 'correlation_matrix_'+ 
            name, header = oids)

        with open(working_dir + name, 'w') as o:
            o.write(bt.delimited_self())

        with open(working_dir + name,'rU') as f:
            samp_ids, bact_names, samp_bact, n_bact, n_samp = \
                parse.otu_parse(f, 'temp.txt', skip = 1, delim = '\t')

        # construct normalized (OTU) table
        otu_table = np.zeros(shape=[n_bact, n_samp])
        for b in xrange(n_bact):
            for s in xrange(n_samp):
                otu_table[b][s] = samp_bact[samp_ids[s]][b]

        otu_table = otu_table/otu_table.sum(axis=0)

        # construct zero-inflated data based on OTU table and renormalize
        zero_infl_otu = np.copy(otu_table) 
        zero_infl_otu[zero_infl_otu<0.001] = 0
        zero_infl_otu = zero_infl_otu/zero_infl_otu.sum(axis=0)
        zero_infl_otu = np.insert(zero_infl_otu, 0, [x for x in range(n_bact)],
            axis = 1)

        # output zero_inflated and non-zero inflated OTU simulated data
        output.print_matrix(zero_infl_otu, working_dir + 'zero_infl_otu_' + 
            name, header = ["#OTU"] + sids)

        otu_table = np.insert(otu_table, 0, [x for x in range(n_bact)],axis = 1)

        output.print_matrix(otu_table, working_dir + 'otu_' + name, 
            header = ["#OTU"] + sids)
    return

if __name__ == "__main__":
    gen_cutie_copula()


