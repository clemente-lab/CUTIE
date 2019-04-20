#!/usr/bin/env python
from __future__ import division

import matplotlib
matplotlib.use('Agg')

import os
import random
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns # ; sns.set(color_codes=True)
from scipy import stats
from decimal import Decimal

from cutie import parse

###
# Creating matrices/dataframes to hold results
###

def print_true_false_corr(initial_corr, true_corr, working_dir, statistic,
                          resample_k, method):
    """
    Prints simplified 2-column table of variable pairs classified as true and
    false correlations for each k in {1...resample_k}
    ----------------------------------------------------------------------------
    INPUTS
    initial_corr - Set of integer tuples. Contains variable pairs initially
                   classified as significant (forward CUtIe) or insignificant
                   (reverse CUtIe). Note variable pairs (i,j) and (j,i) are
                   double counted.
    true_corr    - Dictionary mapping resample index to set of integer tuples.
                   Contains variable pairs classified as true correlations
                   (TP or FN, depending on forward or reverse CUtIe respectively).
    working_dir  - String. File path of working directory specified by user.
    statistic    - String. Analysis being performed.
    resample_k   - Integer. Number of points being resampled by CUtIe.
    method       - String. 'log', 'cbrt' or 'none' depending on method used for
                   evaluating confidence interval (bootstrapping and jackknifing
                   only)
    """
    # function for printing matrix of ses
    def print_sig(corr_set, output_fp):
        matrix = np.zeros(shape=[len(corr_set), 2])
        row = 0
        for point in corr_set:
            matrix[row] = point
            row += 1
        pd.DataFrame(matrix, columns = ['var1', 'var2']).to_csv(output_fp,
            sep = '\t', index = False)

    # iterates through each resampling index
    for k in range(resample_k):
        false_corr = set(initial_corr).difference(set(true_corr[str(k+1)]))
        output_fp = working_dir + 'data_processing/' + statistic + method + \
            str(k+1) + '_falsesig.txt'
        print_sig(false_corr, output_fp)
        output_fp = working_dir + 'data_processing/' + statistic + method + \
            str(k+1) + '_truesig.txt'
        print_sig(true_corr[str(k+1)], output_fp)

def report_results(n_var1, n_var2, working_dir, label, initial_corr, true_corr,
                   true_comb_to_rev, false_comb_to_rev, resample_k, log_fp):
    """
    Writes to log files the number of TP/FP or TN/FN.
    ----------------------------------------------------------------------------
    INPUTS
    n_var1            - Integer. Number of variables in file 1.
    n_var2            - Integer. Number of variables in file 2.
    working_dir       - String. Path of working directory specified by user.
    label             - String. Name of project assigned by user.
    initial_corr      - Set of integer tuples. Contains variable pairs initially
                        classified as significant (forward CUtIe) or
                        insignificant (reverse CUtIe). Note variable pairs (i,j)
                        and (j,i) are double counted.
    true_corr         - Set of integer tuples. Contains variable pairs
                        classified as true correlations (TP or FN, depending on
                        forward or reverse CUtIe respectively).
    true_comb_to_rev  - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation in the set of true_corr (either
                        TP or FN) reverses sign upon removal of a point.
    false_comb_to_rev - Same as true_comb_to_rev but for TN/FP.
    resample_k        - Integer. Number of points being resampled by CUtIe.
    log_fp            - String. File path of log file.
    """

    # helper function for converting and printing dict
    def dict_to_print_matrix(comb_to_rev, fp, i):
        n_pairs = len(comb_to_rev[str(i+1)])
        pairs = np.zeros([n_pairs, 2])
        for p in range(n_pairs):
            pairs[p] = comb_to_rev[str(i+1)][p]
        pd.DataFrame(pairs).to_csv(fp, sep = '\t', index = False)

    # for each resampling value of k
    for i in range(int(resample_k)):
        # write to logs
        write_log('The number of false correlations for ' + str(i+1) + ' is '
                  + str(len(initial_corr)-len(true_corr[str(i+1)])), log_fp)
        write_log('The number of true correlations for ' + str(i+1) + ' is '
                  + str(len(true_corr[str(i+1)])), log_fp)
        # check if reverse sign TP/FN is empty
        if true_comb_to_rev != {}:
            write_log('The number of reversed correlations for TP/FN' + str(i+1)
                      + ' is ' + str(len(true_comb_to_rev[str(i+1)])), log_fp)
            fp = working_dir + 'data_processing/' + 'rev_pairs_TPFN_' + label \
                 + '_resample' + str(i+1) + '.txt'
            dict_to_print_matrix(true_comb_to_rev, fp, i)

        # check if reverse sign FP/TN set is empty
        if false_comb_to_rev != {}:
            write_log('The number of reversed correlations for FP/TN' + str(i+1)
                      + ' is ' + str(len(false_comb_to_rev[str(i+1)])), log_fp)
            fp = working_dir + 'data_processing/' + 'rev_pairs_FPTN_' + label \
                 + '_resample' + str(i+1) + '.txt'
            dict_to_print_matrix(false_comb_to_rev, fp, i)

def generate_pair_matrix(base_regions, regions_set, n_var1, n_var2, working_dir):
    """
    Generate matrix for R where each row is a correlation and each column
    is an indicator value -1 for FP as identified by that metric.
    ----------------------------------------------------------------------------
    INPUTS
    base_regions  - List of strings. Each string describes one group among which
                    the intersections are being computed (such as 'cookd').
    regions_set   - Dictionary. Maps key (region on Venn Diagram) to elements in
                    that set (e.g. variable pairs)
    n_var1        - Integer. Number of variables in file 1.
    n_var2        - Integer. Number of variables in file 2.
    working_dir   - String. Path of working directory as specified by user.
                    Should end in '/'
    """

    headers = ['var1', 'var2']
    for metric in base_regions:
        headers.append(metric)
    pair_matrix = np.zeros([n_var1 * n_var2, len(headers)])

    # initialize the indices of the correlations of the row matrix,
    # each row is a correlation
    for var1 in range(n_var1):
        for var2 in range(n_var2):
            row_number = n_var2 * var1 + var2
            pair_matrix[row_number][0] = var1
            pair_matrix[row_number][1] = var2

    for region in base_regions:
        # region_set is a list of tuples
        region_set = regions_set[region]
        region_index = base_regions.index(region)
        for pair in region_set:
            var1, var2 = pair
            row_number = n_var2 * var1 + var2
            pair_matrix[row_number][region_index + 2] = -1

    pd.DataFrame(pair_matrix, columns = headers).to_csv(working_dir + \
        'data_processing/all_pairs.txt', sep = '\t', index = False)


###
# Graphing
###

def graph_subsets(working_dir, var1_names, var2_names, f1type, f2type, R_matrix,
                  headers, statistic, forward_stats, resample_k, initial_corr,
                  true_corr, false_comb_to_rev, true_comb_to_rev, graph_bound,
                  samp_var1, samp_var2, all_pairs, sim, region_sets,
                  corr_compare, exceeds_points, rev_points, fix_axis):
    """
    Creates folders and plots corresponding to particular sets of variable
    pairs. Pairwise correlation scatterplots are plotted as well as fold p value
    changes. Folders are named <quadrant>_<k>_<n>_<revsign> where quadrant
    refers to TP/FP or TN/FN (if forward or reverse CUtIe was run, respectively)
    k is the number of points in that resampling and revsign is present or
    absent depending if the folder specifically contains reverse sign CUtIes
    or not.
    ----------------------------------------------------------------------------
    INPUTS
    working_dir       - String. Path of working directory specified by user.
    var1_names        - List of strings. List of variables in file 1.
    var2_names        - List of strings. List of variables in file 2.
    f1type            - String. Must be 'map' or 'otu' which specifies parsing
                        functionality to perform on file 1
    f2type            - String. Same as f1type but for file 2.
    Rmatrix           - 2D array. output from print_Rmatrix.
    headers           - List of strings. Refers to column names of Rmatrix.
    statistic         - String. Describes analysis being performed.
    forward_stats     - List of strings. Contains list of statistics e.g. 'kpc'
                        'jkp' that pertain to forward (non-reverse) CUtIe
                        analysis
    resample_k        - Integer. Number of points being resampled by CUtIe.
    initial_corr      - Set of integer tuples. Contains variable pairs initially
                        classified as significant (forward CUtIe) or
                        insignificant (reverse CUtIe). Note variable pairs (i,j)
                        and (j,i) are double counted.
    true_corr         - Set of integer tuples. Contains variable pairs
                        classified as true correlations (TP or FN, depending on
                        forward or reverse CUtIe respectively).
    true_comb_to_rev  - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation in the set of true_corr (either
                        TP or FN) reverses sign upon removal of a point.
    false_comb_to_rev - Dictionary. Same as true_comb_to_rev but for TN/FP.
    graph_bound       - Integer. Upper limit of how many graphs to plot in each
                        set.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    all_pairs         - List of tuples. All variable pairs (i,j and j,i are
                        double counted) are included.
    sim               - Boolean. True if simulated data is used (with known
                        underlying covariances).
    region_sets       - Dictionary. Maps key (region on Venn Diagram) to
                        elements in that set (e.g. variable pairs)
    corr_compare      - Boolean. True if using Cook's D, DFFITS analysis etc.
    exceeds_points    - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point is
                        cutie-ogenic.
    rev_points        - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point
                        induces a sign change.
    fix_axis          - Boolean. True if axes are fixed (max and min for vars).
    """

    # load R_matrix into pandas df
    df_R = pd.DataFrame(R_matrix, columns=headers)

    # generate dataframes in set set for plotting
    dfs, initial_insig_corr, initial_sig_corr = generate_dfs(
        statistic, forward_stats, initial_corr, true_corr, true_comb_to_rev,
        false_comb_to_rev, df_R, resample_k, region_sets, corr_compare,
        all_pairs)

    # plotting below taking in dfs
    plot_dfs(graph_bound, working_dir, f1type, f2type, var1_names,
             var2_names, samp_var1, samp_var2, sim, dfs, initial_insig_corr,
             initial_sig_corr, df_R, exceeds_points, rev_points, fix_axis)

def generate_dfs(statistic, forward_stats, initial_corr, true_corr,
                 true_comb_to_rev, false_comb_to_rev, df_R, resample_k,
                 region_sets, corr_compare, all_pairs):
    """
    Create class object and instances of dataframes corresponding to sets,
    e.g. FP, TP etc.
    ----------------------------------------------------------------------------
    INPUTS
    statistic         - String. Describes analysis being performed.
    forward_stats     - List of strings. Contains list of statistics e.g. 'kpc'
                        'jkp' that pertain to forward (non-reverse) CUtIe
                        analysis.
    df_R              - Pandas dataframe. Contains R_matrix read into df form.
    resample_k        - Integer. Number of points being resampled by CUtIe.
    initial_corr      - Set of integer tuples. Contains variable pairs initially
                        classified as significant (forward CUtIe) or
                        insignificant (reverse CUtIe). Note variable pairs (i,j)
                        and (j,i) are double counted.
    true_corr         - Set of integer tuples. Contains variable pairs
                        classified as true correlations (TP or FN, depending on
                        forward or reverse CUtIe respectively).
    true_comb_to_rev  - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation in the set of true_corr (either
                        TP or FN) reverses sign upon removal of a point.
    false_comb_to_rev - Dictionary. Same as true_comb_to_rev but for TN/FP.
    sim               - Boolean. True if simulated data is used (with known
                        underlying covariances).
    region_sets       - Dictionary. Maps key (region on Venn Diagram) to
                        elements in that set (e.g. variable pairs)
    corr_compare      - Boolean. True if using Cook's D, DFFITS analysis etc.
    all_pairs         - List of tuples. All variable pairs (i,j and j,i are
                        double counted) are included.
    """
    # determine labels depending on forward or reverse cutie
    if statistic in forward_stats:
        true_label, false_label = 'TP', 'FP'
        forward_label = True
    else:
        true_label, false_label = 'FN', 'TN'
        forward_label = False

    # create class for each set of plots
    class df_set:
        # Initializer / Instance Attributes
        def __init__(self, name, pairs, quadrant, rev_sign, rm_subset, k):
            # name is a unique identifier for that set of graphs
            self.name = name
            # pairs is a list of tuples of var pairs in that set
            self.pairs = pairs
            # quadrant is the sector of the grid, i.e. TN/FN/TP/FP
            self.quadrant = quadrant
            # rev sign is a boolean determining whether the DF is tracking
            # reversed sign correlations or not
            self.rev_sign = rev_sign
            # rm_subset is subset of R matrix relevant to that set
            self.rm_subset = rm_subset
            # k is number of resampled points
            self.k = k


    # list of df_sets to go through
    dfs = []
    # dictionary with key = # of points being resampled, entry = set of
    # correlations / variable pairs
    false_corr = {}

    # create N (negatives; TN + FN) and P (positives; TP + FP))
    initial_insig_corr = df_set('initial_insig',
                                set(all_pairs).difference(initial_corr), 'N',
                                False, df_R.loc[df_R['indicators'] == 0], 0)
    initial_sig_corr = df_set('initial_sig', initial_corr, 'P', False,
                              df_R.loc[df_R['indicators'] != 0], 0)

    # create df_set instances
    for i in range(resample_k):
        resample_key = str(i+1)
        # determine non true corrs
        false_corr[resample_key] = \
            set(initial_corr).difference(true_corr[resample_key])
        # create relevant df_sets
        # false_corr is TN or FP
        false_corr_obj = df_set(
            'false_corr', false_corr[resample_key], false_label,
            False, df_R.loc[df_R['indicators'] == -1],
            resample_key)
        false_corr_rev_obj = df_set(
            'false_corr_rev', false_comb_to_rev[resample_key], false_label,
            True, df_R.loc[df_R[false_label + '_rev_indicators'] == 1],
            resample_key)
        # true_corr is either TP or FN
        true_corr_obj = df_set(
            'true_corr', true_corr[resample_key], true_label,
            False, df_R.loc[df_R['indicators'] == 1],
            resample_key)
        true_corr_rev_obj = df_set(
            'true_corr_rev', true_comb_to_rev[resample_key], true_label,
            True, df_R.loc[df_R[true_label + '_rev_indicators'] == 1],
            resample_key)
        # extend dfs list
        dfs.extend([false_corr_obj, false_corr_rev_obj,
                    true_corr_obj, true_corr_rev_obj])

    # if using cook's D, etc.
    if corr_compare:
        for region in region_sets:
            metric_df_TP = df_set(
                region, region_sets[region], true_label, False,
                df_R.loc[df_R[region] == 1], '1')
            metric_df_FP = df_set(
                region, region_sets[region], false_label, False,
                df_R.loc[df_R[region] == -1], '1')
            dfs.extend([metric_df_TP, metric_df_FP])

    return dfs, initial_insig_corr, initial_sig_corr


def plot_dfs(graph_bound, working_dir, f1type, f2type, var1_names,
             var2_names, samp_var1, samp_var2, sim, dfs, initial_insig_corr,
             initial_sig_corr, df_R, exceeds_points, rev_points, fix_axis):
    """
    Plot correlations and distribution of pvalues for each dataframe set.
    ----------------------------------------------------------------------------
    INPUTS
    working_dir       - String. Path of working directory specified by user.
    var1_names        - List of strings. List of variables in file 1.
    var2_names        - List of strings. List of variables in file 2.
    f1type            - String. Must be 'map' or 'otu' which specifies parsing
                        functionality to perform on file 1
    f2type            - String. Same as f1type but for file 2.
    initial_insig_corr- Set of integer tuples. Contains variable pairs initially
                        classified as insignificant.
    initial_sig_corr  - Set of integer tuples. Contains variable pairs initially
                        classified as significant. Note variable pairs (i,j)
                        and (j,i) are double counted.
    true_corr         - Set of integer tuples. Contains variable pairs
                        classified as true correlations (TP or FN, depending on
                        forward or reverse CUtIe respectively).
    true_comb_to_rev  - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation in the set of true_corr (either
                        TP or FN) reverses sign upon removal of a point.
    false_comb_to_rev - Dictionary. Same as true_comb_to_rev but for TN/FP.
    graph_bound       - Integer. Upper limit of how many graphs to plot in each
                        set.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    sim               - Boolean. True if simulated data is used (with known
                        underlying covariances).
    dfs               - List of df_set objects. Each object corresponds to one
                        set of correlations being plotted.
    df_R              - Pandas dataframe. Contains R_matrix read into df form.
    exceeds_points    - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point is
                        cutie-ogenic.
    rev_points        - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point
                        induces a sign change.
    fix_axis          - Boolean. True if axes are fixed (max and min for vars).
    """

    # obtain global max and min for fixed axes
    var1_max, var1_min = np.nanmax(samp_var1), np.nanmin(samp_var1)
    var2_max, var2_min = np.nanmax(samp_var2), np.nanmin(samp_var2)

    # for each relevant set
    for df in dfs:
        # plot random / representative correlations
        plot_corr_sets(graph_bound, df, working_dir, f1type, f2type, var1_names,
                       var2_names, samp_var1, samp_var2, sim, exceeds_points,
                       rev_points, fix_axis, var1_max, var1_min, var2_max,
                       var2_min)

        # this section plots pvalue and fold pvalue change distributions
        plot_pdist(df, working_dir)

    for df in [initial_insig_corr, initial_sig_corr] + dfs:
        # plot simulation corrs (when true corr population parameter is known)
        if sim:
            if len(df.rm_subset['truth']) > 1:
                truths = df.rm_subset['truth']
                truth_fp = working_dir + 'graphs/true_corr_' + df.name + '_' + \
                           df.quadrant + '_' + str(df.k) + '.png'
                title = "True correlation coefficients among %s correlations" \
                        % (df.quadrant)
                plot_figure(truths, truth_fp, df_R, title)

        # plot actual sample values
        if len(df.rm_subset['correlations']) > 1:
            corr_vals = df.rm_subset['correlations']
            corr_vals_fp = working_dir + 'graphs/sample_corr_' + df.name + \
                           '_' + df.quadrant + '_' + str(df.k) + '.png'
            title = "Sample correlation coefficients among %s correlations" \
                    % (df.quadrant)
            plot_figure(corr_vals, corr_vals_fp, df_R, title)


def plot_figure(values, fp, df_R, title):
    """
    Seaborn/matplotlib plotting function
    ----------------------------------------------------------------------------
    INPUTS
    values - np.array. Values being plotted.
    fp     - String. File path in which to save plot.
    df_R   - pd DataFrame. Sets length of y-axis size.
    title  - String. Name of graph.
    """
    values = values[np.isfinite(values)]

    fig = plt.figure()
    try:
        sns.distplot(values, bins=20, kde=False, rug=False)
    except ValueError:
        sns.distplot(values, bins=None, kde=False, rug=False)
    plt.ylim(0, int(len(df_R)/10))
    plt.xlim(-1, 1)
    ax = plt.gca()
    ax.set_title(title, fontsize=10)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    fig.set_tight_layout(True)
    plt.tick_params(axis='both', which='both', top=False, right=False)
    sns.despine()
    plt.savefig(fp)
    plt.close()
    return

def plot_pdist(df, working_dir):
    """
    Helper function for graph_subsets(). Produces plots of p-values and
    fold-pvalue changes for each set of correlations as defined by df.
    ----------------------------------------------------------------------------
    INPUTS
    df          - Dataframe-sets object as constructed in graph_subsets().
    working_dir - String. Path of working directory specified by user.
    """
    # pvalues can be nan
    # fold pvalues can be -Inf/Inf
    pvalues = df.rm_subset['pvalues']
    fold_pvalues = df.rm_subset['p_ratio']

    # hold values for future plotting
    stacked1 = np.stack([pvalues, fold_pvalues, df.rm_subset['avg_var1']], 0)
    stacked2 = np.stack([pvalues, fold_pvalues, df.rm_subset['avg_var2']], 0)

    # compute number of infinity and nan entries
    n_infinite = len(fold_pvalues[~np.isfinite(fold_pvalues)])
    n_nan = len(pvalues[np.isnan(pvalues)])

    pvalues = pvalues[~np.isnan(pvalues)]
    fold_pvalues = fold_pvalues[np.isfinite(fold_pvalues)]

    # construct dataframe
    pvalue_df = pd.DataFrame({
        'pvalues': pvalues,
        'fold_pvalues': fold_pvalues,
        'log_pvalues': df.rm_subset['logpvals'],
        'log_fold_pvalues': np.log(fold_pvalues)})

    dists = ['pvalues', 'fold_pvalues', 'log_pvalues', 'log_fold_pvalues']
    for dist in dists:
        if len(pvalue_df[dist]) > 1:
            stat_vals = pvalue_df[dist]
            stat_vals = stat_vals[~np.isnan(stat_vals)]
            stat_vals = stat_vals[np.isfinite(stat_vals)]
            mean = np.mean(stat_vals)
            std = np.mean(stat_vals)
            additional_title = '; mu = ' + str('%.2E' % Decimal(mean)) + \
                ' std = ' + str('%.2E' % Decimal(std))
            title = 'fold_pvalues; n_infinite = ' + str(n_infinite) + ' ' \
                + 'pvalues; n_nan = ' + str(n_nan) + additional_title
            dist_fp = working_dir + 'graphs/' + df.quadrant + '_' + str(df.k) \
                + '_' + dist + '.png'
            fig = plt.figure()
            sns_plot = sns.distplot(stat_vals, bins=20, kde=False, rug=False)
            plt.tick_params(axis='both', which='both', top=False, right=False)
            sns.despine()
            ax = plt.gca()
            ax.set_title(title, fontsize=10)
            fig.patch.set_visible(False)
            ax.patch.set_visible(False)
            fig.set_tight_layout(True)
            plt.savefig(dist_fp)
            plt.close()

    plot_logp_and_logpfold(df, working_dir, stacked1, 'var1')
    plot_logp_and_logpfold(df, working_dir, stacked2, 'var2')

def plot_logp_and_logpfold(df, working_dir, stacked, var_num):
    """
    Plots logp and logp fold graphs for each variable.
    ----------------------------------------------------------------------------
    INPUTS
    df          - Dataframe-sets object as constructed in graph_subsets().
    Stacked     - np.array. Contains p values, fold p values, and average value
                  of var1 or var2.
    working_dir - String. File path to save files.
    var_num     - String. Labels graph as var1 or var2.
    """
    # Remove correlations with infinite or nan values for their p value or
    # fold p value change
    stacked = stacked[:, np.all(~np.isnan(stacked), axis=0)]
    stacked = stacked[:, np.all(np.isfinite(stacked), axis=0)]

    # obtain log pvalues and log fold change in pvalues and normalized values
    logpvalues = np.log(stacked[0])
    logfoldpvalues = np.log(stacked[1])
    x = (stacked[2])
    # avg value of var ranges widely, so normalizing between 0 and 1 makes
    # plotting easier
    # avg_var = np.cbrt(x * 10 / np.linalg.norm(x))
    avg_var = np.cbrt(x/np.linalg.norm(x))

    new_stacked = np.stack([logpvalues, logfoldpvalues, avg_var], 0)

    df_vals = pd.DataFrame({
        'logp': new_stacked[0],
        'logpfold': new_stacked[1],
        'avg_var': new_stacked[2]
        })

    cmap = sns.cubehelix_palette(as_cmap=True)
    sns.set_style("white")
    title = df.name + '_p_vs_p_fold_' + var_num
    dist_fp = working_dir + 'graphs/' + df.name + '_' + str(df.k) \
              + '_logp_vs_logpfold_' + var_num + '.png'
    fig = plt.figure()
    sns.set_style("white")
    sns.scatterplot(x='logp', y='logpfold', data=df_vals, size='avg_var',
                    sizes=(20, 200), hue='avg_var', palette=cmap, legend=False)
    ax = plt.gca()
    ax.set_title(title, fontsize=10)
    plt.tick_params(axis='both', which='both', top=False, right=False)
    sns.despine()
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)

    # Optionally add a colorbar
    if len(avg_var) != 0:
        cax, _ = matplotlib.colorbar.make_axes(ax)
        normalize = matplotlib.colors.Normalize(vmin=min(avg_var),
                                                vmax=max(avg_var))
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

    #plt.legend(loc='upper left', scatterpoints=1, frameon=False,
    #    title='cbrt norm avg ' + var_num)
    # fig.set_tight_layout(True)
    plt.savefig(dist_fp)
    plt.close()

def plot_corr(row, df_folder_fp, f1type, f2type, var1_names, var2_names,
              samp_var1, samp_var2, sim, resample_k, exceeds_points, rev_points,
              fix_axis, var1_max, var1_min, var2_max, var2_min):
    """
    Helper function for plot_corr_sets(). Plots pairwise correlations within each
    set of correlations as defined by df.
    ----------------------------------------------------------------------------
    INPUTS
    row               - Pandas dataframe row.
    df_folder_fp      - File object. Points to directory where particular set of
                        plots will be stored.
    f1type            - String. Must be 'map' or 'otu' which specifies parsing
                        functionality to perform on file 1
    f2type            - String. Same as f1type but for file 2.
    var1_names        - List of strings. List of variables in file 1.
    var2_names        - List of strings. List of variables in file 2.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    resample_k        - Integer. Number of points being resampled.
    exceeds_points    - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point is
                        cutie-ogenic.
    rev_points        - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point
                        induces a sign change.
    fix_axis          - Boolean. True if axes are fixed (max and min for vars).
    var1_max          - Float. Largest value in var1 to use as upper bound.
    var1_min          - Float. Smallest value in var1 to use as upper bound.
    var2_max          - Float. Largest value in var2 to use as upper bound.
    var2_min          - Float. Smallest value in var2 to use as upper bound.
    """
    var1, var2 = int(row['var1_index']), int(row['var2_index'])

    # obtain variable values
    x = samp_var1[:, var1]
    y = samp_var2[:, var2]
    var1_name = var1_names[var1]
    var2_name = var2_names[var2]

    # shorten var name
    if len(var1_name) > 25:
        var1_name = var1_name[0:25]

    if len(var2_name) > 25:
        var2_name = var2_name[0:25]

    # consolidate variables into pd dataframe
    # example:
    # let cutie = np.array([0,0,0,1,1])
    # let reverse = np.array([0,0,1,0,1])
    # want [0, 0, 0, 1, 2]
    # take c*(c+r)
    pair = var1, var2
    cutie = exceeds_points[str(resample_k)][str(pair)]
    cutie = np.array([1 if z > 0 else 0 for z in cutie])
    reverse = rev_points[str(resample_k)][str(pair)]
    reverse = np.array([1 if z > 0 else 0 for z in reverse])

    cr = cutie*(cutie+reverse)
    pair_df = pd.DataFrame({var1_name:x, var2_name:y, 'cutie/rev': cr})
    pair_df = pair_df.dropna(how='any')

    # create plot and title
    title = 'p, ext_p = ' + '%.2E' % Decimal(row['pvalues']) + \
            ', ' + '%.2E' % Decimal(row['extreme_p']) + ' ' + \
            'Rsq, ext_r2 = ' + '%.2E' % Decimal(row['r2vals']) + \
            ', ' + '%.2E' % Decimal(row['extreme_r'])
    if sim:
        title = title + '_' + 'truth = ' + '%.2E' % Decimal(row['truth'])

    fig = plt.figure()
    sns_plot = sns.lmplot(var1_name, var2_name, data=pair_df, hue='cutie/rev',
                          fit_reg=False)
    if fix_axis:
        sns_plot.set(xlim=(var1_min, var1_max), ylim=(var2_min, var2_max))
    ax = plt.gca()
    ax.set_title(title, fontsize=8)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    fig.set_tight_layout(True)
    plt.tick_params(axis='both', which='both', top=False, right=False)
    sns.despine()
    plt.savefig(df_folder_fp + '/' + str(var1) + '_' + str(var2) + '.png')
    plt.close()

def plot_corr_sets(graph_bound, df, working_dir, f1type, f2type, var1_names,
                   var2_names, samp_var1, samp_var2, sim, exceeds_points,
                   rev_points, fix_axis, var1_max, var1_min, var2_max, var2_min):
    """
    Helper function for graph_subsets(). Plots pairwise correlations within each
    set of correlations as defined by df.
    ----------------------------------------------------------------------------
    INPUTS
    graph_bound       - Integer. Upper limit of how many graphs to plot in each
                        set.
    df                - Dataframe-sets object as constructed in graph_subsets().
    working_dir       - String. Path of working directory specified by user.
    f1type            - String. Must be 'map' or 'otu' which specifies parsing
                        functionality to perform on file 1
    f2type            - String. Same as f1type but for file 2.
    var1_names        - List of strings. List of variables in file 1.
    var2_names        - List of strings. List of variables in file 2.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    exceeds_points    - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point is
                        cutie-ogenic.
    rev_points        - Dict of dict. Outer key is resampling index (k = 1),
                        entry is dict where key is variable pair ('(3,4)') and
                        entry is np array of length n where entry is number of
                        resampling values (from 0 to k) in which that point
                        induces a sign change.
    fix_axis          - Boolean. True if axes are fixed (max and min for vars).
    var1_max          - Float. Largest value in var1 to use as upper bound.
    var1_min          - Float. Smallest value in var1 to use as upper bound.
    var2_max          - Float. Largest value in var2 to use as upper bound.
    var2_min          - Float. Smallest value in var2 to use as upper bound.
    """
    # decide var pairs to plot
    np.random.seed(0)
    if graph_bound <= len(df.rm_subset):
        # samples without replacement
        df_forplot = df.rm_subset.sample(n=graph_bound, replace=False,
                                         weights=None, random_state=42, axis=0)
    else:
        df_forplot = df.rm_subset
    # name and create folder
    df_folder_fp = working_dir + 'graphs/' + df.name + '_' + df.quadrant + '_' \
        + str(df.k) + '_' + str(len(df.pairs))
    if df.rev_sign:
        df_folder_fp = df_folder_fp + '_revsign'
    if os.path.exists(df_folder_fp) is not True:
        os.makedirs(df_folder_fp)

    # plot representative plots
    for index, row in df_forplot.iterrows():
        plot_corr(row, df_folder_fp, f1type, f2type, var1_names,
                  var2_names, samp_var1, samp_var2, sim, df.k, exceeds_points,
                  rev_points, fix_axis, var1_max, var1_min, var2_max, var2_min)

###
# Diagnostic plot handling
###

def diag_plots(samp_counter, var1_counter, var2_counter, resample_k,
               working_dir, paired, samp_var1, samp_var2, n_samp):
    """
    Create diagnostic plots i.e. creates histograms of number of times each
    sample or variable appears in CUtIe's
    ----------------------------------------------------------------------------
    INPUTS
    samp_counter - Dictionary. Key is the index of CUtIe resampling
                   (k = 1, 2, 3, ... etc.) and entry is an array of length
                   n_samp corresponding to how many times the i-th sample
                   appears in CUtIe's when evaluated at resampling = k points)
    var1_counter - Dictionary.  Key is the index of CUtIe resampling
                   (k = 1, 2, 3, ... etc.) and entry is an array of length
                   n_var1 corresponding to how many times the j-th variable
                   appears in CUtIe's when evaluated at resampling = k points)
    var2_counter - Same as var1_counter except for var2.
    resample_k   - Integer. Number of points being resampled by CUtIe.
    working_dir  - String. Path of working directory specified by user.
    paired       - Boolean. True if variables are paired (i.e. file 1 and file
                   2 are the same), False otherwise.
    samp_var1    - 2D array. Each value in row i col j is the level of
                   variable j corresponding to sample i in the order that the
                   samples are presented in samp_ids
    samp_var2    - 2D array. Same as samp_var1 but for file 2.
    n_samp       - Integer. Number of samples.
    """
    diag_stats = ['samp', 'var1', 'var2']
    stats_mapping = {'samp': samp_counter,
                     'var1': var1_counter,
                     'var2': var2_counter}

    # for each diagnostic quantity
    for stats in diag_stats:
        for i in range(resample_k):
            counter = stats_mapping[stats]
            counts = np.zeros(shape=[len(counter[str(i+1)]), 2])
            # create 2D array where col 1 = sample/var index and col 2 =
            # number of times that sample/var appears in CUtIes
            for j in range(len(counter[str(i+1)])):
                counts[j] = np.array([j, counter[str(i+1)][j]])

            pd.DataFrame(counts, columns = ['index', 'count']).to_csv(
                working_dir + 'data_processing/' + 'counter_'  + stats + \
                '_resample' + str(i+1) + '.txt', sep = '\t', index = False)

            # create figure
            fig = plt.figure()
            if stats == 'samp':
                counts_df = pd.DataFrame(
                    {stats:counts[:, 0], 'n_cuties': counts[:, 1]})
                sns_plot = sns.lmplot(stats, 'n_cuties', data=counts_df,
                                      fit_reg=False)
            else:
                counts_df = pd.DataFrame(
                    {stats:counts[:, 0], 'n_cuties': counts[:, 1]})
                sns_plot = sns.lmplot(stats, 'n_cuties', data=counts_df,
                                      fit_reg=False)

            ax = plt.gca()
            fig.patch.set_visible(False)
            ax.patch.set_visible(False)
            diag_fp = working_dir + 'graphs/' + 'counter_' + stats + \
                      '_resample' + str(i+1) + '.png'
            fig.set_tight_layout(True)
            plt.tick_params(axis='both', which='both', top=False, right=False)
            sns.despine()
            plt.savefig(diag_fp)
            plt.close()

###
# Log file handling
###
def init_log(log_dir, defaults_fp, config_fp):
    """
    Initializes log file.
    ----------------------------------------------------------------------------
    INPUTS
    log_dir     - String. Directory where to write log file.
    defaults_fp - String. File path of default configuration file.
    config_fp   - String. File path of configuration on specific runs.

    OUTPUTS
    log_fp      - String. File path of log file output.
    """
    now = datetime.datetime.now()
    log_fp = log_dir + str(now.isoformat()) + '_log.txt'

    # initialize log, write md5 of config files
    with open(log_fp, 'w') as f:
        f.write('Begin logging at ' + str(now.isoformat()))
        f.write('\nThe original command was -df ' + defaults_fp
                + ' -cp ' + config_fp)
        f.write('\nThe defaults_fp config file was '
                + parse.md5Checksum(defaults_fp))
        f.write('\nThe config_fp config file was '
                + parse.md5Checksum(config_fp))

    return log_fp

def write_log(message, log_fp):
    """
    Writes message to log file
    ----------------------------------------------------------------------------
    INPUTS
    log_fp  - String. File path of log file output.
    message - String. Message to write to log.
    """
    with open(log_fp, 'a') as f:
        f.write('\n')
        f.write(message)

    return message
