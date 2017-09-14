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

@click.option('-sm', '--samp_meta_fp', required=True,
              type=click.Path(exists=True),
              help='Input  samp meta file')
@click.option('-sb', '--samp_bact_fp', required=True,
              type=click.Path(exists=True),
              help='Input  samp bact file')
@click.option('-d', '--working_dir', default='', required=False,
              help='Directory to save files')
@click.option('-sc', '--startcol', default=17, required=False,
              type=int, help='starting metabolite col')
@click.option('-ec', '--endcol', default=100, required=False,
              type=int, help='ending metabolite col')
@click.option('-k', '--k', default=1, required=False,
              type=int, help='number to resample')
@click.option('-a', '--alpha', default=0.05, required=False,
              type=float, help='threshold value')
@click.option('--nomc', 'mc', flag_value='nomc',
              default=True, required=False,
              help='True if no mc correction used')
@click.option('--bc', 'mc', flag_value='bc',
              required=False, help='True if using Bonferroni correction')
@click.option('--fwer', 'mc', flag_value='fwer',
              required=False, help='True if using FWER correction')
@click.option('--fdr', 'mc', flag_value='fdr',
              required=False, help='True if using FDR correction')

def lungbactmeta_cutienpc(samp_meta_fp, 
                          samp_bact_fp,
                          working_dir,
                          startcol,
                          endcol,
                          k,
                          alpha,
                          mc):
    """ 
    INPUTS
    samp_meta_fp: file object pointing to samp_meta data file
    samp_bact_fp: file object pointing to samp_bact data file (OTU table)
    working_dir:  file path to where data is processed
    startcol:     int, first column in mapping file holding data
    endcol:       int, ONE AFTER last column in mapping data holding data
    k:            int, number of resamples (k <= sample size-3)
    alpha:        threshold used for alpha, 0.05 by default
    mc:           flag, string used to determine which type of MC to use
                    --nomc is no MC at all, --bc is Bonferroni correction,
                    --fdr is false discovery rate correction

    FUNCTION
    Computes pairwise correlations between each bact, meta pair and 
    for the significant correlations, recomputes correlation for each pair
    after iteratively excluding n observations, differentiating
    true and false correlations on the basis of whether the correlation remains
    significant when each individual observation is dropped
    """
    
    import time
    start_time = time.time()

    ### 
    # Parsing and Pre-processing
    ###

    # create subfolder to hold data analysis files
    if os.path.exists(working_dir + 'data_processing') is not True:
        os.makedirs(working_dir + 'data_processing')
        
    # parse sample-metabolite data table
    # meta file doesn't split on \n but will on \r 
    with open(samp_meta_fp,'rU') as f:    
        samp_ids, meta_names, samp_meta_dict = parse.samp_meta_parse(f, startcol, endcol)
    
    # extract 'L6' or 'L7' label from OTU tables
    label = str(samp_bact_fp).split('_')[-1].split('.')[0] 
    
    # parse OTU table
    with open(samp_bact_fp, 'r') as f:
        bact_names, samp_bact_dict, samp_ids = parse.samp_bact_parse(f)
    
    # special case/hard coded modifications
    exceptions = ['110705RC.N.1.RL']
    for exception in exceptions:
        samp_ids.remove(exception)
        samp_meta_dict.pop(exception, None)
        samp_bact_dict.pop(exception, None)
        print 'Removed subject ID ' + str(exception)
    
    n_bact = len(bact_names)
    n_meta = len(meta_names)
    n_samp = len(samp_ids)

    print 'The length of samp_ids is ' + str(n_samp)
    print 'The length of metabolite_names is ' + str(n_meta)
    print 'The length of bact_names is ' + str(n_bact)

    samp_bact_matrix = parse.dict_to_matrix(samp_bact_dict, samp_ids)
    samp_meta_matrix = parse.dict_to_matrix(samp_meta_dict, samp_ids)

    ###
    # Initial Statistics
    ### 

    if k > n_samp - 3:
        print 'Too many points specified for resampling for size ' + str(len(n_samp))
    
    functions = ['stats.linregress']
    mapf = {'stats.linregress': stats.linregress}
    f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','ppvalue','stderr']}
    
    
    '''
    functions = ['stats.linregress', 'stats.spearmanr']
    mapf = {'stats.linregress': stats.linregress,
            'stats.spearmanr': stats.spearmanr}
    f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','ppvalue','stderr'],
               'stats.spearmanr':
                   ['scorr','spvalue']}
    '''

    stat_dict = statistics.initial_stats_SLR(samp_ids, 
                                             samp_bact_matrix, 
                                             samp_meta_matrix, 
                                             functions,
                                             mapf,
                                             f_stats)
    
    output.print_stats_SLR(stat_dict,
                           working_dir,
                           label,
                           functions,
                           mapf,
                           f_stats)
    
    pvalue_matrix = stat_dict['stats.linregress'][3]
    corr_matrix = stat_dict['stats.linregress'][2]
    logp_matrix = np.log(stat_dict['stats.linregress'][3])
    r2_matrix = np.square(stat_dict['stats.linregress'][2])
    
    ### 
    # Update Statistics
    ### 

    print 'The type of mc correction used was ' + mc

    threshold = statistics.set_threshold(pvalue_matrix, alpha, mc)

    print 'The threshold value was ' + str(threshold)
     
    n_meta = np.size(samp_meta_matrix,1)
    n_bact = np.size(samp_bact_matrix,1)
    n_samp = len(samp_ids)
    n_corr = n_meta * n_bact
    
    # create lists of points
    SLR_initial_sig = []
    SLR_true_sig = {}
    rev_dict = {} 
    for i in xrange(n_samp):
        SLR_true_sig[str(i+1)] = []
        rev_dict[str(i+1)] = []

    # for each bact, meta pair
    for bact in xrange(n_bact): 
        for meta in xrange(n_meta): 
            point = (bact,meta)
            if pvalue_matrix[bact][meta] < threshold and pvalue_matrix[bact][meta] != 0.0:
                SLR_initial_sig.append(point)
                exceeds = np.zeros(n_samp)
                sign = np.sign(corr_matrix[bact][meta])
                for i in xrange(k):
                    reverse, delta = statistics.resamplek_cutie_pc(bact, 
                                                          meta, 
                                                          samp_ids, 
                                                          samp_bact_matrix, 
                                                          samp_meta_matrix,
                                                          threshold,
                                                          i+1,
                                                          sign)
                    exceeds = np.add(exceeds, delta)
                # sums to 0
                    if exceeds.sum() == 0:
                        SLR_true_sig[str(i+1)].append(point)
                        if reverse.sum() != 0:
                            rev_dict[str(i+1)].append(point)
    ###                       
    # Report results and write output files
    ###

    for i in xrange(k):
        print 'The number of false correlations for ' + str(i+1) + ' is ' + str(
            len(SLR_initial_sig)-len(SLR_true_sig[str(i+1)])) 
        print 'The number of true correlations for ' + str(i+1) + ' is ' + str(
            len(SLR_true_sig[str(i+1)]))
        print 'The number of reversed correlations for ' + str(i+1) + ' is ' + str(
            len(rev_dict[str(i+1)]))


        sig_indicators = statistics.indicator(n_bact,
                                              n_meta,
                                              SLR_initial_sig,
                                              SLR_true_sig[str(i+1)])
        
        output.print_matrix(sig_indicators, 
                            working_dir + 'data_processing/sig_indicators_' + label 
                            + '_resample' + str(i+1) + '.txt',
                            ['bact_index','meta_index'])
        
        pairs = rev_dict[str(i+1)]
        n_pairs = len(pairs)
        pair_matrix = np.zeros([n_pairs,2])
        for p in xrange(n_pairs):
            pair_matrix[p] = pairs[p]
        output.print_matrix(pair_matrix, 
                            working_dir + 'data_processing/' + 'rev_pairs_' + label 
                            + '_resample' + str(i+1) + '.txt',
                            ['bact_index', 'meta_index'])


    
    ###
    # Matrices for R analysis
    ###

    # Retrieve mean bacterial abundances and levels of metabolites
    avg_bact_matrix = np.array([np.mean(samp_bact_matrix,0)])
    avg_meta_matrix = np.array([np.mean(samp_meta_matrix,0)]) 

    output.print_matrix(avg_bact_matrix, 
                        working_dir + 'data_processing/bact_avg_' + label + '.txt',
                        bact_names)
    output.print_matrix(avg_meta_matrix, 
                        working_dir + 'data_processing/meta_avg_' + label + '.txt',
                        meta_names)


    entries = [avg_bact_matrix, 
               avg_meta_matrix, 
               pvalue_matrix, 
               logp_matrix, 
               r2_matrix, 
               sig_indicators, # add k
               n_bact,
               n_meta]

    headers = ['avg_bact',
               'avg_meta',
               'pvals',
               'logp',
               'r2vals',
               'indicators',
               'bact_index',
               'meta_index']

    R_matrix = np.zeros([n_corr, len(entries)])
    row = 0
    for b in xrange(n_bact):
        for m in xrange(n_meta):
            R_matrix[row][0] = avg_bact_matrix[0][b]
            R_matrix[row][1] = avg_meta_matrix[0][m]
            R_matrix[row][2] = pvalue_matrix[b][m]
            R_matrix[row][3] = logp_matrix[b][m]
            R_matrix[row][4] = r2_matrix[b][m]
            R_matrix[row][5] = sig_indicators[b][m] # right now it only works for k = 1
            R_matrix[row][6] = b
            R_matrix[row][7] = m
            row += 1

    output.print_matrix(R_matrix, 
                        working_dir + 'data_processing/R_matrix_' + label + '.txt',
                        headers)

    print time.time() - start_time
    return

if __name__ == "__main__":
    lungbactmeta_cutienpc()
