#!/usr/bin/env python
import os
import datetime
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns # ; sns.set(color_codes=True)
from cutie import parse
from cutie import utils

###
# Creating matrices/dataframes to hold results
###

def report_results(initial_corr, true_corr,
                   correct_to_rev, incorrect_to_rev, resample_k, log_fp):
    """
    Writes to log files the number of TP/FP or TN/FN.
    ----------------------------------------------------------------------------
    INPUTS
    working_dir       - String. Path of working directory specified by user.
    initial_corr      - Set of integer tuples. Contains variable pairs initially
                        classified as significant (forward CUtIe) or
                        insignificant (reverse CUtIe). Note variable pairs (i,j)
                        and (j,i) are double counted.
    true_corr         - Set of integer tuples. Contains variable pairs
                        classified as true correlations (TP or FN, depending on
                        forward or reverse CUtIe respectively).
    correct_to_rev    - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation in the set of true_corr (either
                        TP or FN) reverses sign upon removal of a point.
    incorrect_to_rev  - Same as true_comb_to_rev but for TN/FP.
    resample_k        - Integer. Number of points being resampled by CUtIe.
    log_fp            - String. File path of log file.
    """
    # for each resampling value of k
    for i in range(int(resample_k)):
        # write to logs
        write_log('The number of false correlations for ' + str(i+1) + ' is '
                  + str(len(initial_corr)-len(true_corr[str(i+1)])), log_fp)
        write_log('The number of true correlations for ' + str(i+1) + ' is '
                  + str(len(true_corr[str(i+1)])), log_fp)

        # check if reverse sign TP/FN is empty
        if correct_to_rev != {}:
            write_log('The number of reversed correlations for TP/FN' + str(i+1)
                      + ' is ' + str(len(correct_to_rev[str(i+1)])), log_fp)

        # check if reverse sign FP/TN set is empty
        if incorrect_to_rev != {}:
            write_log('The number of reversed correlations for FP/TN' + str(i+1)
                      + ' is ' + str(len(incorrect_to_rev[str(i+1)])), log_fp)

def print_summary_df(n_var1, n_var2,
                     col_names, col_vars, working_dir, resample_index, n_corr,
                     paired=False):
    """
    Creates summary datafrane containing CUtIe's analysis results.
    Each row is a correlation and columns contain relevant statistics e.g.
    pvalue, correlation strength etc.
    ----------------------------------------------------------------------------
    INPUTS
    n_var1         - Integer. Number of variables in file 1.
    n_var2         - Integer. Number of variables in file 2.
    col_names      - List of strings. Contains names of columns (e.g. pvalues).
    col_vars       - List of 2D arrays. Contains various statistics (e.g. 2D
                     array of pvalues, 2D array of correlations). For each
                     array, the entry in i-th row, j-th column contains the
                     value of that particular statistic for the correlation
                     between variable i and j (i in file 1, j in file 2).
    resample_index - String (cast from int). Number of points being resampled by
                     CUTIE.
    n_corr         - Number of correlations performed by CUTIE. If variables are
                     paired, n_corr = (n choose 2) * 2 as correlations are
                     double counted (only corr(i,i) are ignored)
    paired         - Boolean. True if variables are paired (i.e. file 1 and file
                     2 are the same), False otherwise.
    OUTPUTS
    summary_df     - Array. Dataframe object summarizing the above statistics
                     and features per correlation (variable pair).
    """
    # create header row
    headers = ['var1_index', 'var2_index']

    for var in col_names:
        headers.append(var)

    # create matrix locally in python
    summary_matrix = np.zeros([n_corr, len(headers)])
    row = 0
    for var1 in range(n_var1):
        for var2 in range(n_var2):
            if not (paired and (var1 <= var2)):
                entries = [var1, var2]
                for col_var in col_vars:
                    entries.append(col_var[var1][var2])
                summary_matrix[row] = np.array([entries])
                row += 1

    # convert to dataframe
    summary_df = pd.DataFrame(summary_matrix, columns=headers)

    summary_df.to_csv(working_dir + 'data_processing/summary_df_resample_' + \
        str(resample_index) + '.txt', sep='\t', index=False)

    return summary_df

###
# Graphing
###

def graph_subsets(working_dir, var1_names, var2_names, f1type, f2type, summary_df,
                  statistic, forward_stats, resample_k, initial_corr, true_corr,
                  true_corr_to_rev, false_corr_to_rev, graph_bound, samp_var1,
                  samp_var2, all_pairs, region_sets, corr_compare, exceeds_points,
                  rev_points, fix_axis):
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
    summary_df        - Dataframe. Output from print_summary_df.
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
    true_corr_to_rev  - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation in the set of true_corr (either
                        TP or FN) reverses sign upon removal of a point.
    false_corr_to_rev - Dictionary. Same as true_corr_to_rev but for TN/FP.
    graph_bound       - Integer. Upper limit of how many graphs to plot in each
                        set.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    all_pairs         - List of tuples. All variable pairs (i,j and j,i are
                        double counted) are included.
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

    # generate dataframes in set set for plotting
    dfs, initial_insig_corr, initial_sig_corr = generate_dfs(
        statistic, forward_stats, initial_corr, true_corr, true_corr_to_rev,
        false_corr_to_rev, summary_df, resample_k, region_sets, corr_compare,
        all_pairs)

    if statistic in forward_stats:
        forward = True
    else:
        forward = False

    # plotting below taking in dfs
    plot_dfs(graph_bound, working_dir, f1type, f2type, var1_names, var2_names,
             samp_var1, samp_var2, dfs, initial_insig_corr, initial_sig_corr,
             summary_df, exceeds_points, rev_points, fix_axis, forward)

def generate_dfs(statistic, forward_stats, initial_corr, true_corr,
                 true_corr_to_rev, false_corr_to_rev, summary_df, resample_k,
                 region_sets, corr_compare, all_pairs):
    """
    Create class object and instances of dataframes corresponding to sets,
    e.g. FP, TP etc.
    ----------------------------------------------------------------------------
    INPUTS
    statistic         - String. Describes analysis being performed.
    forward_stats     - List of strings. Contains list of statistics e.g. 'pearson'
                        'spearman' that pertain to forward (non-reverse) CUTIE
                        analysis.
    summary_df        - Pandas dataframe. Summary_df from print_summary_df.
    resample_k        - Integer. Number of points being resampled by CUtIe.
    initial_corr      - Set of integer tuples. Contains variable pairs initially
                        classified as significant (forward CUtIe) or
                        insignificant (reverse CUtIe). Note variable pairs (i,j)
                        and (j,i) are double counted.
    true_corr         - Set of integer tuples. Contains variable pairs
                        classified as true correlations (TP or FN, depending on
                        forward or reverse CUtIe respectively).
    true_corr_to_rev  - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation that was correctly classified
                        (TP or TN) reverses sign upon removal of a point.
    false_corr_to_rev - Dictionary. Same as true_corr_to_rev but for FP/FN.
    summary_df        - Pandas dataframe. Contains summary_df from print_summary_df.
    region_sets       - Dictionary. Maps key (region on Venn Diagram) to
                        elements in that set (e.g. variable pairs)
    corr_compare      - Boolean. True if using Cook's D, DFFITS analysis etc.
    all_pairs         - List of tuples. All variable pairs (i,j and j,i are
                        double counted) are included.
    """
    # determine labels depending on forward or reverse cutie
    if statistic in forward_stats:
        true_label, false_label = 'TP', 'FP'
    else:
        true_label, false_label = 'FN', 'TN'

    # create class for each set of plots
    class dfSet:
        # Initializer / Instance Attributes
        def __init__(self, name, pairs, quadrant, rev_sign, sm_subset, k):
            # name is a unique identifier for that set of graphs
            self.name = name
            # pairs is a list of tuples of var pairs in that set
            self.pairs = pairs
            # quadrant is the sector of the grid, i.e. TN/FN/TP/FP
            self.quadrant = quadrant
            # rev sign is a boolean determining whether the DF is tracking
            # reversed sign correlations or not
            self.rev_sign = rev_sign
            # sm_subset is subset of summary matrix relevant to that set
            self.sm_subset = sm_subset
            # k is number of resampled points
            self.k = k


    # list of df_sets to go through
    dfs = []
    # dictionary with key = # of points being resampled, entry = set of
    # correlations / variable pairs
    false_corr = {}

    # create N (negatives; TN + FN) and P (positives; TP + FP))
    initial_insig_corr = dfSet('initial_insig',
                               set(all_pairs).difference(initial_corr), 'N',
                               False, summary_df.loc[summary_df['indicators'] == 0], 0)
    initial_sig_corr = dfSet('initial_sig', initial_corr, 'P', False,
                             summary_df.loc[summary_df['indicators'] != 0], 0)

    # create df_set instances
    for i in range(resample_k):
        resample_key = str(i+1)
        # determine non true corrs
        false_corr[resample_key] = \
            set(initial_corr).difference(true_corr[resample_key])
        # create relevant df_sets
        # false_corr are FPs or TNs
        false_corr_obj = dfSet(
            'false_corr', false_corr[resample_key], false_label,
            False, summary_df.loc[summary_df['indicators'] == -1],
            resample_key)
        false_corr_rev_obj = dfSet(
            'false_corr_rev', false_corr_to_rev[resample_key], false_label,
            True, summary_df.loc[summary_df[false_label + '_rev_indicators'] == 1],
            resample_key)
        # true_corr is either TP or FN
        true_corr_obj = dfSet(
            'true_corr', true_corr[resample_key], true_label,
            False, summary_df.loc[summary_df['indicators'] == 1],
            resample_key)
        true_corr_rev_obj = dfSet(
            'true_corr_rev', true_corr_to_rev[resample_key], true_label,
            True, summary_df.loc[summary_df[true_label + '_rev_indicators'] == 1],
            resample_key)
        # extend dfs list
        dfs.extend([false_corr_obj, false_corr_rev_obj,
                    true_corr_obj, true_corr_rev_obj])

    # if using cook's D, etc.
    if corr_compare:
        for region in region_sets:
            TP_metric_df = dfSet(
                region, region_sets[region], true_label, False,
                summary_df.loc[summary_df[region] == 1], '1')
            FP_metric_df = dfSet(
                region, region_sets[region], false_label, False,
                summary_df.loc[summary_df[region] == -1], '1')
            dfs.extend([TP_metric_df, FP_metric_df])

    return dfs, initial_insig_corr, initial_sig_corr


def plot_dfs(graph_bound, working_dir, f1type, f2type, var1_names, var2_names,
             samp_var1, samp_var2, dfs, initial_insig_corr, initial_sig_corr,
             summary_df, exceeds_points, rev_points, fix_axis, forward):
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
    dfs               - List of df_set objects. Each object corresponds to one
                        set of correlations being plotted.
    summary_df        - Pandas dataframe. Contains summary_df from print_summary_df.
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
    forward           - Boolean. True if CUTIE is run in the forward direction, False if
                        reverse.
    """

    # obtain global max and min for fixed axes
    var1_max, var1_min = np.nanmax(samp_var1), np.nanmin(samp_var1)
    var2_max, var2_min = np.nanmax(samp_var2), np.nanmin(samp_var2)

    # for each relevant set
    for df in dfs:
        # plot random / representative correlations
        plot_corr_sets(graph_bound, df, working_dir, f1type, f2type, var1_names,
                       var2_names, samp_var1, samp_var2, exceeds_points,
                       rev_points, fix_axis, var1_max, var1_min, var2_max,
                       var2_min, forward)

        # this section plots pvalue and fold pvalue change distributions
        plot_pdist(df, working_dir)

    for df in [initial_insig_corr, initial_sig_corr] + dfs:
        # plot actual sample values
        if len(df.sm_subset['correlations']) > 1:
            corr_vals = df.sm_subset['correlations']
            corr_vals_fp = working_dir + 'graphs/sample_corr_' + df.name + \
                           '_' + df.quadrant + '_' + str(df.k) + '.png'
            title = "Sample correlation coefficients among %s correlations" \
                    % (df.quadrant)
            plot_figure(corr_vals, corr_vals_fp, summary_df, title)


def plot_figure(values, fp, summary_df, title):
    """
    Seaborn/matplotlib plotting function
    ----------------------------------------------------------------------------
    INPUTS
    values     - np.array. Values being plotted.
    fp         - String. File path in which to save plot.
    summary_df - Pandas dataframe. Contains summary_df from print_summary_df.
                 Sets length of y-axis size.
    title      - String. Name of graph.
    """
    values = values[np.isfinite(values)]

    fig = plt.figure()
    try:
        sns.distplot(values, bins=20, kde=False, rug=False)
    except ValueError:
        sns.distplot(values, bins=None, kde=False, rug=False)
    plt.ylim(0, int(len(summary_df)/10))
    plt.xlim(-1, 1)
    ax = plt.gca()
    ax.set_title(title, fontsize=10)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    fig.set_tight_layout(True)
    plt.tick_params(axis='both', which='both', top=False, right=False)
    sns.despine()
    plt.savefig(fp)
    plt.close('all')

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
    pvalues = df.sm_subset['pvalues']
    fold_pvalues = df.sm_subset['p_ratio']

    # compute number of infinity and nan entries
    n_infinite = len(fold_pvalues[~np.isfinite(fold_pvalues)])
    n_nan = len(pvalues[np.isnan(pvalues)])

    pvalues = pvalues[~np.isnan(pvalues)]
    fold_pvalues = fold_pvalues[np.isfinite(fold_pvalues)]

    # construct dataframe
    pvalue_df = pd.DataFrame({
        'pvalues': pvalues,
        'fold_pvalues': fold_pvalues,
        'log_pvalues': np.log(pvalues),
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
            sns.distplot(stat_vals, bins=20, kde=False, rug=False)
            plt.tick_params(axis='both', which='both', top=False, right=False)
            sns.despine()
            ax = plt.gca()
            ax.set_title(title, fontsize=10)
            fig.patch.set_visible(False)
            ax.patch.set_visible(False)
            fig.set_tight_layout(True)
            plt.savefig(dist_fp)
            plt.close('all')

def plot_corr(row, df_folder_fp, var1_names, var2_names, samp_var1, samp_var2,
              resample_k, exceeds_points, rev_points, fix_axis, var1_max,
              var1_min, var2_max, var2_min, forward):
    """
    Helper function for plot_corr_sets(). Plots pairwise correlations within each
    set of correlations as defined by df.
    ----------------------------------------------------------------------------
    INPUTS
    row               - Pandas dataframe row.
    df_folder_fp      - File object. Points to directory where particular set of
                        plots will be stored.
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
    forward           - Boolean. True if CUTIE is run in the forward direction, False if
                        reverse.
    """
    var1, var2 = int(row['var1_index']), int(row['var2_index'])

    # obtain variable values
    x = samp_var1[:, var1]
    y = samp_var2[:, var2]
    var1_name = var1_names[var1]
    var2_name = var2_names[var2]

    # convert variable name of otu formats
    if var1_name[0:3] == 'k__':
        var1_name = utils.read_taxa(var1_name)

    if var2_name[0:3] == 'k__':
        var2_name = utils.read_taxa(var2_name)


    # shorten var name
    if len(var1_name) > 25:
        var1_name = var1_name[0:25]

    if len(var2_name) > 25:
        var2_name = var2_name[0:25]

    # consolidate variables into pd dataframe
    # example:
    # let cutie = np.array([0,0,0,1,1])
    # this indicates that samples 4 and 5 were contributing to either FP or FN status
    # if forward is true, then we want to investigate TP with sign changes
    # so we take cutie = 1 - cutie, obtaining [1, 1, 1, 0 ,0]
    # let reverse = np.array([1,0,1,0,1])
    # want [2, 1, 2, 0, 0]
    # take c*(c+r)
    # it matters if it is forward or backward; need correct corr
    pair = var1, var2
    cutie = exceeds_points[str(resample_k)][str(pair)]
    cutie = np.array([1 if z > 0 else 0 for z in cutie])
    if forward:
        cutie = 1 - cutie
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
    plt.close('all')

def plot_corr_sets(graph_bound, df, working_dir, f1type, f2type, var1_names,
                   var2_names, samp_var1, samp_var2, exceeds_points, rev_points,
                   fix_axis, var1_max, var1_min, var2_max, var2_min, forward):
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
    forward           - Boolean. True if CUTIE is run in the forward direction, False if
                        reverse.
    """
    # decide var pairs to plot
    np.random.seed(0)
    if graph_bound <= len(df.sm_subset):
        # samples without replacement
        df_forplot = df.sm_subset.sample(n=graph_bound, replace=False,
                                         weights=None, random_state=42, axis=0)
    else:
        df_forplot = df.sm_subset
    # name and create folder
    df_folder_fp = working_dir + 'graphs/' + df.name + '_' + df.quadrant + '_' \
        + str(df.k) + '_' + str(len(df.pairs))
    if df.rev_sign:
        df_folder_fp = df_folder_fp + '_revsign'
    if os.path.exists(df_folder_fp) is not True:
        os.makedirs(df_folder_fp)

    # plot representative plots
    for index, row in df_forplot.iterrows():
        plot_corr(row, df_folder_fp, var1_names,
                  var2_names, samp_var1, samp_var2, df.k, exceeds_points,
                  rev_points, fix_axis, var1_max, var1_min, var2_max, var2_min,
                  forward)

###
# Diagnostic plot handling
###

def diag_plots(samp_counter, var1_counter, var2_counter, resample_k, working_dir,
               paired):
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
    paired       - Boolean. True if variables are paired (same in both files).
    """
    if paired:
        diag_stats = ['samp', 'var']
        var_counter = {}
        for i in range(resample_k):
            var_counter[str(i+1)] = var1_counter[str(i+1)] + var2_counter[str(i+1)]
        stats_mapping = {'samp': samp_counter,
                         'var': var_counter}
    else:
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

            pd.DataFrame(counts, columns=['index', 'count']).to_csv(
                working_dir + 'data_processing/counter_'  + stats + \
                '_resample' + str(i+1) + '.txt', sep='\t', index=False)

            # create figure
            fig = plt.figure()
            if stats == 'samp':
                counts_df = pd.DataFrame(
                    {stats:counts[:, 0], 'n_cuties': counts[:, 1]})
                sns.lmplot(stats, 'n_cuties', data=counts_df,
                           fit_reg=False)
            else:
                counts_df = pd.DataFrame(
                    {stats:counts[:, 0], 'n_cuties': counts[:, 1]})
                sns.lmplot(stats, 'n_cuties', data=counts_df,
                           fit_reg=False)

            ax = plt.gca()
            fig.patch.set_visible(False)
            ax.patch.set_visible(False)
            diag_fp = working_dir + 'graphs/counter_' + stats + \
                      '_resample' + str(i+1) + '.png'
            fig.set_tight_layout(True)
            plt.tick_params(axis='both', which='both', top=False, right=False)
            sns.despine()
            plt.savefig(diag_fp)
            plt.close('all')

###
# Log file handling
###
def init_log(log_dir, input_config_fp):
    """
    Initializes log file.
    ----------------------------------------------------------------------------
    INPUTS
    log_dir         - String. Directory where to write log file.
    input_config_fp - String. File path of configuration on specific runs.

    OUTPUTS
    log_fp      - String. File path of log file output.
    """
    now = datetime.datetime.now()
    log_fp = log_dir + str(now.isoformat()) + '_log.txt'

    # initialize log, write md5 of config files
    with open(log_fp, 'w') as f:
        f.write('Begin logging at ' + str(now.isoformat()))
        f.write('\nThe original command was -i ' + input_config_fp)
        f.write('\nThe input_config_fp md5 was ' + parse.md5_checksum(input_config_fp))

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
