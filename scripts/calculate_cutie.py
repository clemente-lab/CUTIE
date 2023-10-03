#!/usr/bin/env python
from cutie import parse
from cutie import output
from cutie import utils
from cutie import statistics
from cutie import __version__

import matplotlib
matplotlib.use('Agg')

import time
import click
import shutil
import numpy as np
import pandas as pd
import datetime
import sys
from pathlib import Path

import random
from decimal import Decimal

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy import stats


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)

@click.option('-i', '--input_config_fp', required=False,
              type=click.Path(exists=True),
              help='Input config file path')


def calculate_cutie(input_config_fp):
    """
    Computes pairwise correlations between each variable pair and
    for the significant correlations, recomputes correlation for each pair
    after iteratively excluding n observations, differentiating
    true and false correlations on the basis of whether the correlation remains
    significant when each individual observation is dropped
    """
    # unpack config variables
    (samp_var1_fp, delimiter1, samp_var2_fp, delimiter2, f1type, f2type,
     working_dir, skip1, skip2, startcol1, endcol1, startcol2, endcol2, param,
     statistic, corr_compare, resample_k, paired, overwrite, alpha, multi_corr,
     fold, fold_value, graph_bound, fix_axis) = parse.parse_config(input_config_fp)

    # create working directory
    if Path(working_dir).is_dir() is not True:
        print('No working directory found, creating working_dir...')
        Path(working_dir).mkdir()
    elif overwrite is not True:
        print('Working directory already exists, exiting...')
        sys.exit()
    else: # if overwrite is True and working dir already exists
        print('Overwriting pre-existing directory...')
        shutil.rmtree(working_dir)
        Path(working_dir).mkdir()

    if Path(working_dir + 'data_processing').is_dir() is not True:
        Path(working_dir + 'data_processing').mkdir()

    # initialize and write log file
    start_time = time.process_time()
    log_fp = output.init_log(working_dir, input_config_fp)

    ###
    # Parsing and Pre-processing
    ###
    print('Parsing input...')
    # define possible stats
    forward_stats = ['pearson',  'spearman', 'kendall', 'mine']
    reverse_stats = ['rpearson', 'rspearman', 'rkendall', 'rmine']
    all_stats = forward_stats + reverse_stats
    if statistic in forward_stats:
        forward = True
    elif statistic in reverse_stats:
        forward = False

    if statistic not in all_stats:
        raise ValueError('Invalid statistic: %s chosen' % statistic)
    if corr_compare and resample_k != 1:
        raise ValueError('Resample_k must be 1 for pointwise stats')

    # file handling and parsing decisions
    # file 1 is the 'dominant' file type and should always contain the OTU file
    # we let the dominant fil 'override' the sample_id list ordering
    samp_ids2, var2_names, samp_var2_df, n_var2, n_samp = parse.parse_input(
        f2type, samp_var2_fp, startcol2, endcol2, delimiter2, skip2)
    output.write_log('The length of variables for file 2 is ' + str(n_var2), log_fp)
    output.write_log('The number of samples for file 2 is ' + str(n_samp), log_fp)
    output.write_log('The md5 of samp_var2 was ' + \
        str(parse.md5_checksum(samp_var2_fp)), log_fp)

    samp_ids1, var1_names, samp_var1_df, n_var1, n_samp = parse.parse_input(
        f1type, samp_var1_fp, startcol1, endcol1, delimiter1, skip1)
    output.write_log('The length of variables for file 1 is ' + str(n_var1), log_fp)
    output.write_log('The number of samples for file 1 is ' + str(n_samp), log_fp)
    output.write_log('The md5 of samp_var1 was ' + \
        str(parse.md5_checksum(samp_var1_fp)), log_fp)

    # if the samp_ids differ, only take common elements
    samp_ids = [value for value in samp_ids1 if value in samp_ids2]
    n_samp = len(samp_ids)

    # subset dataframe, obtain avg and variance
    samp_var1 = parse.process_df(samp_var1_df, samp_ids)
    samp_var2 = parse.process_df(samp_var2_df, samp_ids)

    # printing of samp and var names for reference
    output.write_log('There are ' + str(len(samp_ids)) + ' samples', log_fp)
    output.write_log('The first 3 samples are ' + str(samp_ids[0:3]), log_fp)
    if len(var1_names) >= 3:
        output.write_log('The first 3 var1 are ' + str(var1_names[0:3]), log_fp)
    else:
        output.write_log('Var1 was ' + str(var1_names), log_fp)
    if len(var2_names) >= 3:
        output.write_log('The first 3 var2 are ' + str(var2_names[0:3]), log_fp)
    else:
        output.write_log('Var2 was ' + str(var2_names), log_fp)

    ###
    # Pearson, Spearman, Kendall
    ###

    print('Computing initial (pre-CUTIE) significance of all pairwise correlations...')

    # initial setup
    pvalues, corrs, r2vals = statistics.initial_stats(samp_var1,
        samp_var2, statistic, paired)

    # determine parameter (either r or p)
    output.write_log('The parameter chosen was ' + param, log_fp)

    # determine statistic
    output.write_log('The statistic chosen was ' + statistic, log_fp)

    # determine significance threshold and number of correlations
    if param == 'p':
        output.write_log('The type of mc correction used was ' + multi_corr, log_fp)
    threshold, n_corr, minp = statistics.set_threshold(pvalues, param, alpha,
                                                       multi_corr, paired)
    output.write_log('The threshold value was ' + str(threshold), log_fp)

    # calculate initial sig candidates
    initial_corr, all_pairs = statistics.get_initial_corr(n_var1, n_var2,
        pvalues, corrs, threshold, param, paired)

    output.write_log('The number of correlations is ' + str(len(all_pairs)),
        log_fp)

    # change initial_corr if doing rCUtIe
    if statistic in reverse_stats:
        initial_corr = set(all_pairs).difference(initial_corr)

    output.write_log('The length of initial_corr is ' + str(len(initial_corr)),
        log_fp)

    # if interested in evaluating dffits, dsr, etc.
    region_sets = []
    if corr_compare:
        print('Assessing CooksD, DFFITS, DSR...')
        infln_metrics = ['cutie_1pc', 'cookd', 'dffits', 'dsr']
        infln_mapping = {
            'cutie_1pc': statistics.resample1_cutie_pc,
            'cookd': statistics.cookd,
            'dffits': statistics.dffits,
            'dsr': statistics.dsr
        }
        (incorrect_infln_sets, region_combs, region_sets) = statistics.pointwise_comparison(
            infln_metrics, infln_mapping, samp_var1, samp_var2, initial_corr,
            threshold, fold_value, fold, param, forward)

        for region in region_combs:
            output.write_log('The amount of unique elements in set ' +
                             str(region) + ' is ' +
                             str(len(region_sets[str(region)])), log_fp)

        # report results
        for metric in infln_metrics:
            metric_incorrect = incorrect_infln_sets[metric]
            if forward:
                false_corrs = len(metric_incorrect)
                true_corrs = len(initial_corr) - len(metric_incorrect)
            else:
                false_corrs = len(initial_corr) - len(metric_incorrect)
                true_corrs = len(metric_incorrect)

            output.write_log('The number of false correlations according to ' +
                             metric + ' is ' + str(false_corrs), log_fp)
            output.write_log('The number of true correlations according to ' +
                             metric + ' is ' + str(true_corrs), log_fp)

    # return sets of interest; some of these will be empty dicts depending
    # on the statistic
    print('Performing CUTIE resampling...')

    (true_corr, true_corr_to_rev, false_corr_to_rev, corr_extrema_p,
    corr_extrema_r, samp_counter, var1_counter,
    var2_counter, exceeds_points, rev_points) = statistics.update_cutiek_true_corr(
        initial_corr, samp_var1, samp_var2, pvalues, corrs, threshold,
        statistic, forward_stats, reverse_stats, resample_k, fold, fold_value, param)


    ###
    # Determine indicator matrices
    ###

    # element i,j is -1 if flagged by CUtIe as FP, 1 if TP,
    # and 0 if insig originally
    true_indicators = utils.return_indicators(n_var1, n_var2, initial_corr,
                                        true_corr, resample_k)

    true_rev_indicators = utils.return_indicators(n_var1, n_var2,
        initial_corr, true_corr_to_rev, resample_k)

    false_rev_indicators = utils.return_indicators(n_var1, n_var2,
        initial_corr, false_corr_to_rev, resample_k)

    if corr_compare:
        metric_set_to_indicator = {}
        for region in region_sets:
            region_truths = {}
            if forward:
                region_truths['1'] = set(initial_corr).difference(region_sets[region])
            else:
                region_truths['1'] = region_sets[region]
            metric_set_to_indicator[region] = utils.return_indicators(
                n_var1, n_var2, initial_corr, region_truths, 1)['1']


    ###
    # Report statistics
    ###

    print('Printing results...')

    for k in range(resample_k):
        resample_key = str(k+1)

        # for Spearman and MIC, R2 value stored is same as rho or MIC
        # respectively
        p_ratio = np.divide(corr_extrema_p[resample_key], pvalues)
        r2_ratio = np.divide(np.square(corr_extrema_r[resample_key]), r2vals)
        stat_values = [pvalues, corrs, r2vals,
            true_indicators[resample_key], true_rev_indicators[resample_key],
            false_rev_indicators[resample_key], corr_extrema_p[resample_key],
            corr_extrema_r[resample_key], p_ratio, r2_ratio]
        if statistic in forward_stats:
            stat_names = ['pvalues', 'correlations', 'r2vals',
                'indicators','TP_rev_indicators', 'FP_rev_indicators',
                'extreme_p', 'extreme_r', 'p_ratio', 'r2_ratio']
        elif statistic in reverse_stats:
            stat_names = ['pvalues', 'correlations', 'r2vals',
                'indicators', 'FN_rev_indicators', 'TN_rev_indicators',
                'extreme_p', 'extreme_r', 'p_ratio', 'r2_ratio']

        # for pointwise
        if corr_compare:
            stat_names.extend(region_sets)
            for region in region_sets:
                stat_values.append(metric_set_to_indicator[region])

        # Output results, write summary df
        summary_df = output.print_summary_df(var1_names, var2_names,
                stat_names, stat_values, working_dir, resample_key, n_corr,
                paired, forward)


        output.report_results(initial_corr, true_corr, true_corr_to_rev,
                              false_corr_to_rev, resample_key, log_fp)

    ###
    # Graphing
    ###

    # create subfolder to hold graphing files
    if Path(working_dir + 'graphs').is_dir() is not True:
        Path(working_dir + 'graphs').mkdir()

    output.graph_subsets(working_dir, var1_names, var2_names, f1type, f2type,
        summary_df, statistic, forward_stats, resample_k, initial_corr,
        true_corr, true_corr_to_rev, false_corr_to_rev, graph_bound, samp_var1,
        samp_var2, all_pairs, region_sets, corr_compare, exceeds_points,
        rev_points, fix_axis)

    output.diag_plots(samp_ids, var1_names, var2_names, samp_counter, var1_counter,
        var2_counter, resample_k, working_dir, paired, statistic, forward_stats)

    # write log file
    output.write_log('The runtime was ' + str(time.process_time() - start_time), log_fp)
    now = datetime.datetime.now()
    output.write_log('Ended logging at ' + str(now.isoformat()), log_fp)

    return

if __name__ == "__main__":
    calculate_cutie()
