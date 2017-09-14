#!/usr/bin/env python
from __future__ import division

from cutie import parse
from cutie import statistics
from cutie import output
from cutie import __version__

import click
import os
import numpy as np
from scipy import stats

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)

@click.option('-b', '--samp_bact_fp', required=True,
              type=click.Path(exists=True),
              help='Input  samp bact file')
@click.option('-d', '--working_dir', default='', required=False,
              help='Directory to save files')
@click.option('-k', '--k', default=1, required=False,
              type=int, help='number to resample')



def lungbact_cutiekprop(samp_bact_fp,
                        working_dir,
                        k):
    """ 
    INPUTS
    samp_bact_fp: file object pointing to samp_bact data file (OTU table)
    working_dir:  file path to where data is processed

    FUNCTION
    """
    
    import time
    start_time = time.time()

    ### 
    # Parsing and Pre-processing
    ###
    # create subfolder to hold data analysis files
    if os.path.exists(working_dir + 'data_processing') is not True:
        os.makedirs(working_dir + 'data_processing')
    
    # extract 'L6' or 'L7' label from OTU tables
    label = str(samp_bact_fp).split('_')[-1].split('.')[0] 
    
    # parse OTU table
    with open(samp_bact_fp, 'r') as f:
        bact_names, samp_bact_dict, samp_ids = parse.samp_bact_parse(f)
    
    print 'The length of samp_ids is ' + str(len(samp_ids))
    print 'The length of bact_names is ' + str(len(bact_names))

    samp_bact_matrix = parse.dict_to_matrix(samp_bact_dict, samp_ids)
    
    ###
    # Generate Log and Var Log Matrices and correct for zeros
    ###

    samp_bact_mr, samp_bact_clr, samp_bact_lclr, samp_bact_varlog, correction, n_zero = statistics.multi_zeros(samp_bact_matrix)
    print 'The number of non-zero entries is ' + str(n_zero) + ' out of ' + str(np.size(samp_bact_matrix))
    print 'The zero correction was /2, ' + str(correction)
    
    ###
    # Initial Proportionality Statistics
    ### 

    n_samp = len(samp_ids)
    n_bact = np.size(samp_bact_matrix,1)
    n_corr = n_bact * (n_bact - 1)/2

    prop = statistics.initial_stats_prop(samp_bact_clr, samp_bact_varlog)
    header = [str(x+1) for x in xrange(n_bact)]
    output.print_matrix(prop, working_dir + 'data_processing/prop.txt', header)
    output.print_matrix(samp_bact_mr, working_dir + 'data_processing/samp_bact_mr.txt', header)

    ### 
    # Update Proportionality Statistics
    ### 

    prop_threshold = 0.05

    # create lists of points
    prop_initial_sig = []
    prop_true_sig = {}
    for i in xrange(k):
        prop_true_sig[str(i+1)] = []

    for bact1 in xrange(n_bact): 
        for bact2 in xrange(n_bact): 
            if bact1 == bact2:
                prop[bact1][bact2] == 0.0
            point = (bact1,bact2)
            if prop[bact1][bact2] < prop_threshold and prop[bact1][bact2] != 0.0:
                prop_initial_sig.append(point)
                exceeds = np.zeros(n_samp)

                for i in xrange(k): 
                    delta = statistics.resamplek_cutie_prop(bact1, 
                                                            bact2, 
                                                            samp_bact_clr, 
                                                            prop_threshold,
                                                            k)
                    exceeds = np.add(exceeds, delta)

                    # sums to 0
                    if exceeds.sum() == 0:
                        prop_true_sig[str(i+1)].append(point)

    for i in xrange(k):
        print 'The number of false correlations for ' + str(i+1) + ' is ' + str(
            len(prop_initial_sig)-len(prop_true_sig[str(i+1)])) 
        print 'The number of true correlations for ' + str(i+1) + ' is ' + str(
            len(prop_true_sig[str(i+1)]))


        indicators = statistics.indicator(n_bact,
                                          n_bact,
                                          prop_initial_sig,
                                          prop_true_sig[str(i+1)])
        
        output.print_matrix(indicators, 
                            working_dir + 'data_processing/sig_indicators_cutie_prop' + label 
                            + '_resample' + str(i+1) + '.txt')


    ###
    # Matrices for R analysis
    ###

    # Retrieve mean bacterial abundances and levels of metabolites
    avg_bact_mr_matrix = np.array([np.mean(samp_bact_mr,0)])

    output.print_matrix(avg_bact_mr_matrix, 
                            working_dir + 'data_processing/bact_avg_mr_' + label + '.txt')


    headers = ['avg_bact_mr_1',
               'avg_bact_mr_2',
               'indicators',
               'prop',
               'bact1_index',
               'bact2_index']

    R_matrix = np.zeros([int(n_corr), len(headers)])
    row = 0
    for bact1 in xrange(n_bact):
        for bact2 in xrange(bact1):
            R_matrix[row][0] = avg_bact_mr_matrix[0][bact1]
            R_matrix[row][1] = avg_bact_mr_matrix[0][bact2]
            R_matrix[row][2] = indicators[bact1][bact2]
            R_matrix[row][3] = prop[bact1][bact2]
            R_matrix[row][4] = bact1
            R_matrix[row][5] = bact2
            row += 1

    output.print_matrix(R_matrix, 
                        working_dir + 'data_processing/R_matrix_' + label + '.txt',
                        headers)


    print time.time() - start_time
    return

if __name__ == "__main__":
    lungbact_cutiekprop()
