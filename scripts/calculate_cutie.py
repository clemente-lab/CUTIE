#!/usr/bin/env python
from __future__ import division

from cutie import parse
from cutie import statistics
from cutie import output
from cutie import __version__

import minepy
import time
import click
import os
import numpy as np
import pandas as pd
import datetime

import random
from decimal import Decimal
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy import stats


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)

@click.option('-df', '--defaults_fp', required=False,
              type=click.Path(exists=True),
              help='Input  default config file path')
@click.option('-cf', '--config_fp', required=False,
              type=click.Path(exists=True),
              help='Input  config file path')


def calculate_cutie(defaults_fp, config_fp):
    """ 
    Computes pairwise correlations between each bact, meta pair and 
    for the significant correlations, recomputes correlation for each pair
    after iteratively excluding n observations, differentiating
    true and false correlations on the basis of whether the correlation remains
    significant when each individual observation is dropped
    """
    # unpack config variables
    (label, samp_var1_fp, delimiter1, samp_var2_fp, delimiter2, f1type, f2type, 
    mine_fp, minep_fp, pskip, mine_delimiter, working_dir, skip1, skip2, 
    startcol1, endcol1, startcol2, endcol2, statistic, corr_compare, resample_k,
    paired, alpha, mc, fold, fold_value, n_replicates, log_transform1, 
    log_transform2, CI_method, sim, corr_path, graph_bound, 
    log_dir) = parse.parse_config(defaults_fp, config_fp)

    # initialize and write log file
    start_time = time.clock()
    log_fp = output.init_log(log_dir, defaults_fp, config_fp)

    ### 
    # Parsing and Pre-processing
    ###

    # define possible stats
    forward_stats = ['kpc', 'jkp', 'bsp', 'ksc', 'jks', 
                     'bss', 'mine', 'jkm', 'bsm']
    reverse_stats = ['rkpc', 'rjkp', 'rbsp', 'rsc', 'rjks', 
                     'rbss', 'rmine', 'rjkm', 'rbsm']
    all_stats = forward_stats + reverse_stats
    pearson_stats = ['kpc', 'jkp' , 'bsp', 'rpc', 'rjkp', 'rbsp']
    spearman_stats = ['ksc', 'jks','bss', 'rsc', 'rjks', 'rbss']
    mine_stats = ['mine', 'rmine', 'jkm', 'bsm', 'rjkm', 'rbsm']
    if statistic not in all_stats:
        raise ValueError('Invalid statistic: %s chosen' % statistic)
        
    # create subfolder to hold data analysis files
    if os.path.exists(working_dir + 'data_processing') is not True:
        os.makedirs(working_dir + 'data_processing')
        
    # file handling and parsing decisions
    # file 1 is the 'dominant' file type and should always contain the OTU file
    # we let the dominant fil 'override' the sample_id list ordering
    samp_ids, var2_names, samp_to_var2, n_var2, n_samp = \
        parse.parse_input(f2type, samp_var2_fp, startcol2, endcol2, delimiter2, 
                          skip2, log_fp)
    output.write_log('The md5 of samp_var2 was ' + \
        str(parse.md5Checksum(samp_var2_fp)), log_fp)   
    samp_ids, var1_names, samp_to_var1, n_var1, n_samp = \
        parse.parse_input(f1type, samp_var1_fp, startcol1, endcol1, delimiter1, 
                          skip1, log_fp)
    output.write_log('The md5 of samp_var1 was ' + \
        str(parse.md5Checksum(samp_var1_fp)), log_fp)   

    # printing of samp and var names for reference
    output.write_log('The first 5 samples are ' + str(samp_ids[0:5]), log_fp)
    output.write_log('The first 5 var1 are ' + str(var1_names[0:5]), log_fp)
    output.write_log('The first 5 var2 are ' + str(var2_names[0:5]), log_fp)

    # convert dictionaries to matrices
    samp_var1, avg_var1, norm_avg_var1, var_var1, norm_var_var1, skew_var1 = \
        parse.dict_to_matrix(samp_to_var1, samp_ids)    
    samp_var2, avg_var2, norm_avg_var2, var_var2, norm_var_var2, skew_var2 = \
        parse.dict_to_matrix(samp_to_var2, samp_ids)

    # log transform of data (if log_transform1 or log_transform2 are true)
    if log_transform1:
        samp_var1 = statistics.log_transform(samp_var1, working_dir, 1)
        output.write_log('Variable 1 was log-transformed')
    if log_transform2:
        samp_var2 = statistics.log_transform(samp_var2, working_dir, 2)
        output.write_log('Variable 2 was log-transformed')

    ###
    # Pearson, Spearman, and MIC
    ### 
    # pull mine-specific data
    if statistic in mine_stats:
        # obtain p_value bins
        with open(minep_fp, 'rU') as f:
            mine_bins, pvalue_bins = parse.parse_minep(f, mine_delimiter, pskip)
    else:
        # placeholder variables
        mine_bins = np.nan
        pvalue_bins = np.nan

    # statistic-specific initial output
    stat_to_matrix = statistics.assign_statistics(samp_var1, samp_var2, 
        statistic, pearson_stats, spearman_stats, mine_stats, mine_bins,
        pvalue_bins, paired, f1type, log_fp)

    # unpack statistic matrices
    pvalues = stat_to_matrix['pvalues']
    corrs = stat_to_matrix['correlations']
    logpvals = stat_to_matrix['logpvals']
    r2vals = stat_to_matrix['r2vals']

    # determine significance threshold and number of correlations
    threshold, n_corr = statistics.set_threshold(pvalues, alpha, mc, log_fp, 
                                                 paired)

    # calculate initial sig candidates
    initial_corr, all_pairs = statistics.get_initial_corr(n_var1, n_var2, 
        pvalues, threshold, paired, log_fp)

    # return sets of interest; some of these will be empty dicts depending 
    # on the statistic
    (true_corr, true_comb_to_rev, false_comb_to_rev, corr_extrema_p, 
    corr_extrema_r, samp_counter, var1_counter, 
    var2_counter) = statistics.updatek_cutie(initial_corr, pvalues, samp_var1, 
        samp_var2, threshold, resample_k, corrs, fold, fold_value, working_dir, 
        CI_method, forward_stats, reverse_stats, pvalue_bins, mine_bins, paired,
        statistic, n_replicates)

    # if interested in evaluating dffits, dsr, etc.
    if corr_compare:
        statistics.pointwise_comparison(samp_var1, samp_var2, pvalues, corrs,
            working_dir, n_corr, initial_corr, threshold, statistic, fold_value, 
            log_fp, paired, fold)


    ###
    # Determine indicator matrix of significance
    ###

    # element i,j is -1 if flagged by CUtIe as FP, 1 if TP, 
    # and 0 if insig originally
    true_indicators = statistics.return_indicators(n_var1, n_var2, initial_corr, 
                                        true_corr, resample_k)

    true_rev_indicators = statistics.return_indicators(n_var1, n_var2, 
        initial_corr, true_comb_to_rev, resample_k)

    false_rev_indicators = statistics.return_indicators(n_var1, n_var2, 
        initial_corr, false_comb_to_rev, resample_k)

    ###
    # Report statistics
    ###

    for k in xrange(resample_k):
        resample_key = str(k+1)

        # for Spearman and MIC, R2 value stored is same as rho or MIC 
        # respectively
        p_ratio = np.divide(corr_extrema_p[resample_key], pvalues)
        r2_ratio = np.divide(corr_extrema_r[resample_key], r2vals)
        variables = [pvalues, logpvals, corrs, r2vals,
            true_indicators[resample_key], true_rev_indicators[resample_key],
            false_rev_indicators[resample_key], corr_extrema_p[resample_key], 
            corr_extrema_r[resample_key], p_ratio, r2_ratio]
        if statistic in forward_stats:
            variable_names = ['pvalues', 'logpvals', 'correlations', 'r2vals', 
                'indicators','TP_rev_indicators', 'FP_rev_indicators',
                'extreme_p', 'extreme_r', 'p_ratio', 'r2_ratio']
        elif statistic in reverse_stats:
            variable_names = ['pvalues', 'logpvals', 'correlations', 'r2vals', 
                'indicators', 'FN_rev_indicators', 'TN_rev_indicators', 
                'extreme_p', 'extreme_r', 'p_ratio', 'r2_ratio']

        # for simulations only
        if sim == True:
            corr_values = np.loadtxt(corr_path, usecols=range(n_var1), 
                comments="#", skiprows = 1, delimiter="\t", unpack=False)
            variable_names.append('truth')
            variables.append(corr_values)

        # Output results, write R matrix
        if statistic in forward_stats:
            output.report_results(n_var1, n_var2, working_dir, label,
                                  initial_corr, true_corr, true_comb_to_rev, 
                                  false_comb_to_rev, resample_key, log_fp)

            R_matrix, headers = output.print_Rmatrix(avg_var1, avg_var2, 
                var_var1, var_var2, skew_var1, 
                skew_var2, n_var1, n_var2, variable_names, variables, 
                working_dir, resample_key, label, n_corr, statistic, paired)

            # print pairs of false_sig and true_sig (for create_json.py)
            output.print_true_false_corr(initial_corr, true_corr, working_dir, 
                statistic, resample_k, CI_method)

        elif statistic in reverse_stats:
            output.report_results(n_var1, n_var2, working_dir, label,
                                  initial_corr, true_corr, true_comb_to_rev, 
                                  false_comb_to_rev, resample_key)

            R_matrix, headers = output.print_Rmatrix(avg_var1, avg_var2, 
                var_var1, var_var2, skew_var1, 
                skew_var2, n_var1, n_var2, variable_names, variables, 
                working_dir, resample_key, label + 'rev', n_corr, statistic, 
                paired)

            # print pairs of false_sig and true_sig(for create_json.py)
            output.print_true_false_corr(initial_corr, true_corr, working_dir, 
                statistic, resample_k, CI_method)

    ###
    # Graphing
    ###

    output.graph_subsets(working_dir, var1_names, var2_names, f1type, f2type, 
        R_matrix, headers, statistic, forward_stats, 
        resample_k, initial_corr, true_corr, false_comb_to_rev, 
        true_comb_to_rev, graph_bound, samp_var1, samp_var2)

    # output histograms showing sample and variable appearance among CUtIes
    lof = statistics.lof_fit(samp_var1, samp_var2, n_samp, working_dir, paired, 
                             log_fp)

    output.diag_plots(samp_counter, var1_counter, var2_counter, resample_k, 
        working_dir, paired, samp_var1, samp_var2, n_samp, lof)

    # write log file
    output.write_log('The runtime was ' + str(time.clock() - start_time), log_fp)
    now = datetime.datetime.now()
    output.write_log('Ended logging at ' + str(now.isoformat()), log_fp)

    return

if __name__ == "__main__":
    calculate_cutie()
