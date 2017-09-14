#!/usr/bin/env python
from __future__ import division

from cutie import parse
from cutie import statistics
from cutie import output
from cutie import __version__

import click
import os
import itertools
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

def lungbactmeta_cutiepc(samp_meta_fp, 
                         samp_bact_fp,
                         working_dir,
                         startcol,
                         endcol,
                         alpha,
                         mc):
    """ 
    INPUTS
    samp_meta_fp: file object pointing to samp_meta data file
    samp_bact_fp: file object pointing to samp_bact data file (OTU table)
    working_dir:  file path to where data is processed
    startcol:     int, first column in mapping file holding data
    endcol:       int, ONE AFTER last column in mapping data holding data
    alpha:        threshold used for alpha, 0.05 by default
    mc:           flag, string used to determine which type of MC to use
                    --nomc is no MC at all, --bc is Bonferroni correction,
                    --fdr is false discovery rate correction

    FUNCTION
    Computes pairwise correlations between each bact, meta pair and 
    for the significant Pearson correlations, recomputes correlation for each pair
    after iteratively excluding individual observations, differentiating
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

    ### 
    # Update Statistics
    ### 

    print 'The type of mc correction used was ' + mc

    threshold = statistics.set_threshold(pvalue_matrix, alpha, mc)

    print 'The threshold value was ' + str(threshold)


    infln_metrics = ['cutie_1pc', 'cookd', 'dffits', 'dsr']
    infln_mapping = {
                    'cutie_1pc': statistics.resample1_cutie_pc,
                    'cookd': statistics.cookd,
                    'dffits': statistics.dffits,
                    'dsr': statistics.dsr
                    }

    SLR_initial_sig, true_sig, infln_dict = statistics.update_cutie(
                                                samp_ids, 
                                                samp_bact_matrix,
                                                samp_meta_matrix, 
                                                pvalue_matrix,
                                                infln_metrics,
                                                infln_mapping,
                                                threshold)
    
    '''
    if os.path.exists(working_dir + 'corr_metrics') is not True:
        os.makedirs(working_dir + 'corr_metrics')
    '''
    # initialize data structures
    # point_dict is a dict where the key is a given combination of metrics and each entry
    # is a particular set of bact, meta, sample
    # corr_dict is a dict where each key is a given combination of metrics and each entry
    # is a bact, meta correlation 
    
    point_dict = {}
    corr_dict = {}
    combs = []
    for i in xrange(1, len(infln_metrics)+1):
        els = [list(x) for x in itertools.combinations(infln_metrics, i)]
        combs.extend(els)

    for comb in combs:
        point_dict[str(comb)] = []
        corr_dict[str(comb)] = []

    # 
    for point in SLR_initial_sig:
        b,m = point
        for s in xrange(n_samp):
            for comb in combs: # [1], [2, 3], etc.
                infln = True
                for c in comb: # each individual key     
                    if infln_dict[c][s][b][m] == 0:
                        infln = False
                        break
                if infln is True:
                    point = b, m, s, samp_bact_matrix[s][b], samp_meta_matrix[s][m], pvalue_matrix[b][m]
                    point_dict[str(comb)].append(point)

    for point in SLR_initial_sig:
        b, m = point
        for comb in combs: # [1], [2, 3], etc.
            FP = True
            for c in comb: # each individual key    
                if sum(infln_dict[c],0)[b][m] == 0:
                    FP = False
                    break
            if FP is True:
                corr_dict[str(comb)].append(point)
                
    for comb in combs:
        print 'The amount of influential points in set ' + str(comb) + ' is ' + str(len(point_dict[str(comb)]))
        points = point_dict[str(comb)]
        n_points = len(points)
        point_matrix = np.zeros([n_points,6])
        for p in xrange(n_points):
            point_matrix[p] = points[p]
        output.print_matrix(point_matrix, working_dir + 'data_processing/' + 'points_' + label + '_' + str(comb) + '.txt')
            
    for comb in combs:
        print 'The amount of FP in set ' + str(comb) + ' is ' + str(len(corr_dict[str(comb)]))
        pairs = corr_dict[str(comb)]
        n_pairs = len(pairs)
        pair_matrix = np.zeros([n_pairs,2])
        for p in xrange(n_pairs):
            pair_matrix[p] = pairs[p]
        output.print_matrix(pair_matrix, working_dir + 'data_processing/' + 'pairs_' + label + '_' + str(comb) + '.txt')
        

        
    ###
    # Matrices for R analysis
    ###

    # Retrieve mean bacterial abundances and levels of metabolites
    avg_bact_matrix = np.array([np.mean(samp_bact_matrix,0)])
    avg_meta_matrix = np.array([np.mean(samp_meta_matrix,0)]) 

    output.print_matrix(avg_bact_matrix, 
                            working_dir + 'data_processing/bact_avg_' + label + '.txt')
    output.print_matrix(avg_meta_matrix, 
                            working_dir + 'data_processing/meta_avg_' + label + '.txt')

    for metric in infln_metrics:
        SLR_true_sig = true_sig[metric]
        print 'The number of false correlations according to ' + metric + ' is ' + str(len(SLR_initial_sig)-len(SLR_true_sig)) 
        print 'The number of true correlations according to ' + metric + ' is ' + str(len(SLR_true_sig))
            
        indicators = statistics.indicator(n_bact,
                                          n_meta,
                                          SLR_initial_sig,
                                          SLR_true_sig)

        output.print_matrix(indicators, 
                                working_dir + 'data_processing/sig_indicators_' + label + '_' + metric + '.txt')

    print time.time() - start_time
    return

if __name__ == "__main__":
    lungbactmeta_cutiepc()
