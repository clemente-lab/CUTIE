#!/usr/bin/env python
from __future__ import division
    
import os
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy import stats
from decimal import Decimal            

from cutie import parse

###
# Creating matrices/dataframes to hold results
###

def print_matrix(matrix, output_fp, header, delimiter = '\t'):
    """
    Creates .txt file (or desired output format) of a 2D matrix. Open will 
    remove any pre-existing file with the same name.
    ----------------------------------------------------------------------------
    INPUTS
    matrix    - Array. 2D array to be printed.
    output_fp - String. Path and name of destination of file. 
    delimiter - String. Delimiter used when printing file.
    header    - List of strings. Contains names of each column. Length must 
                match number of cols of matrix.
    """
    # obtain dimensions
    rows = np.size(matrix, 0)
    cols = np.size(matrix, 1)

    with open(output_fp, 'w') as f:
        # write header
        for h in header:
            f.write(h + delimiter)
        f.write('\n')
        # write rows and columns
        for r in xrange(rows):
            for c in xrange(cols):
                f.write(str(matrix[r][c]) + delimiter)
            f.write('\n')


def print_Rmatrix(avg_var1, avg_var2, var_var1, var_var2, skew_var1, skew_var2, 
    n_var1, n_var2, col_names, col_vars, working_dir, resample_index, label, 
    n_corr, statistic = 'kpc', paired = False):
    """
    Creates dataframe easily loaded into R containing CUtIe's analysis results. 
    Each row is a correlation and columns contain relevant statistics e.g. 
    pvalue, correlation strength etc.
    ----------------------------------------------------------------------------
    INPUTS
    avg_var1       - 1D array where k-th entry is mean value for variable k. 
                     Variables are ordered as in original data file (i.e. order 
                     is presered through parsing) for file 1.
    avg_var2       - 1D array. Same as avg_var1 but for file 2.
    var_var1       - 1D array where k-th entry is unbiased variance for variable
                     k for file 1. 
    var_var2       - Same as var_var1 but for file 2.
    skew_var1      - 1D array where k-th entry is skew of variable k for file 1.
    skew_var2      - Same as skew_var1 but for file 2.
    n_var1         - Integer. Number of variables in file 1.
    n_var2         - Integer. Number of variables in file 2.
    col_names      - List of strings. Contains names of columns (e.g. pvalues).
    col_vars       - List of 2D arrays. Contains various statistics (e.g. 2D 
                     array of pvalues, 2D array of correlations). For each 
                     array, the entry in i-th row, j-th column contains the 
                     value of that particular statistic for the correlation 
                     between variable i and j (i in file 1, j in file 2).
    resample_index - Integer. Number of points being resampled by CUtIe.
    label          - String. Name of project assigned by user.
    n_corr         - Number of correlations performed by CUtIe. If variables are 
                     paired, n_corr = (n choose 2) * 2 as correlations are 
                     double counted (only corr(i,i) are ignored)
    statistic      - String. Describes type of analysis.
    paired         - Boolean. True if variables are paired (i.e. file 1 and file
                     2 are the same), False otherwise.

    OUTPUTS
    Rmatrix        - Array. 2D array/dataframe-like object easily loaded into R
                     summarizing above variables per correlation.
    headers        - List of strings. Refers to column names of Rmatrix.
    """
    # create header row
    headers = ['var1_index','var2_index','avg_var1','avg_var2',
                'var_var1','var_var2','skew_var1','skew_var2']

    for var in col_names:
        headers.append(var)

    # create matrix locally in python
    R_matrix = np.zeros([n_corr, len(headers)])
    row = 0
    for var1 in xrange(n_var1):
        for var2 in xrange(n_var2):
            if not (paired and (var1 == var2)):
                entries = [var1, var2, 
                            avg_var1[0][var1], 
                            avg_var2[0][var2], 
                            var_var1[0][var1], 
                            var_var2[0][var2], 
                            skew_var1[0][var1],
                            skew_var2[0][var2]]
                for col_var in col_vars:
                    entries.append(col_var[var1][var2])
                R_matrix[row] = np.array([entries])
                row += 1

    print_matrix(R_matrix, working_dir + 'data_processing/R_matrix_' + label + \
                        '_resample_' + resample_index + '.txt', headers, '\t')

    return R_matrix, headers

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
    true_corr    - Set of integer tuples. Contains variable pairs classified as 
                   true correlations (TP or FN, depending on forward or reverse 
                   CUtIe respectively).
    working_dir  - String. File path of working directory specified by user.
    statistic    - String. Analysis being performed.
    resample_k   - Integer. Number of points being resampled by CUtIe.
    method       - String. 'log', 'cbrt' or 'none' depending on method used for 
                   evaluating confidence interval (bootstrapping and jackknifing 
                   only)
    """
    # function for printing matrix of ses
    def print_sig(corr_set, output_fp):
        matrix = np.zeros(shape=[len(corr_set),2])
        row = 0
        for point in corr_set:
            matrix[row] = point
            row += 1
        print_matrix(matrix, output_fp,  header = \
            ['var1','var2'], delimiter = '\t')

    # iterates through each resampling index
    for k in xrange(resample_k):
        false_corr = set(initial_corr).difference(set(true_corr[str(k+1)]))
        output_fp = working_dir + 'data_processing/' + statistic + method + \
            str(k+1) + '_falsesig.txt'
        print_sig(false_corr, output_fp)
        output_fp = working_dir + 'data_processing/' + statistic + method + \
            str(k+1) + '_truesig.txt'
        print_sig(true_corr, output_fp)
     
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
        pairs = np.zeros([n_pairs,2])
        for p in xrange(n_pairs):
            pairs[p] = comb_to_rev[str(i+1)][p]
        print_matrix(pairs, fp, '\t')

    # for each resampling value of k
    for i in xrange(int(resample_k)):
        # write to logs
        write_log('The number of false correlations for ' + str(i+1) + ' is ' 
            + str(len(initial_corr)-len(true_corr[str(i+1)])), log_fp) 
        write_log('The number of true correlations for ' + str(i+1) + ' is ' 
            + str(len(true_corr[str(i+1)])), log_fp)
        # check if reverse sign TP/FN is empty 
        if true_comb_to_rev != {}:
            write_log('The number of reversed correlations for TP/FN' + str(i+1) 
                + ' is ' + str(len(comb_to_rev1[str(i+1)])), log_fp)
            fp = working_dir + 'data_processing/' + 'rev_pairs_TPFN_' + label \
                + '_resample' + str(i+1) + '.txt'
            dict_to_print_matrix(true_comb_to_rev, fp, i)

        # check if reverse sign FP/TN set is empty
        if false_comb_to_rev != {}:
            write_log('The number of reversed correlations for FP/TN' + str(i+1)
                + ' is ' + str(len(false_comb_to_rev[str(i+1)])), log_fp)
            fp = working_dir + 'data_processing/' + 'rev_pairs_FPTN_' + label \
                + '_resample' + str(i+1) + '.txt'
            dict_to_print_matrix(true_comb_to_rev, fp, i)

###
# Graphing
###

def graph_subsets(working_dir, var1_names, var2_names, f1type, f2type, R_matrix, 
    headers, statistic, forward_stats, 
    resample_k, initial_corr, true_corr, false_comb_to_rev, 
    true_comb_to_rev, graph_bound, samp_var1, samp_var2):
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
    """

    # create subfolder to hold graphing files
    if os.path.exists(working_dir + 'graphs') is not True:
        os.makedirs(working_dir + 'graphs')

    # load R_matrix into pandas df
    df = pd.DataFrame(R_matrix, columns = headers)

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
        def __init__(self, pairs, quadrant, rev_sign, rm_subset, k):
            self.pairs = pairs
            self.quadrant = quadrant
            self.rev_sign = rev_sign
            self.rm_subset = rm_subset
            self.k = k

    dfs = []
    false_corr = {}

    # create df_set instances
    for i in xrange(resample_k):
        resample_key = str(i+1)
        # determine non true corrs
        false_corr[resample_key] = \
            set(initial_corr).difference(true_corr[resample_key])
        # create relevant df_sets
        false_corr_obj = df_set(false_corr[resample_key], false_label, False, 
            df.loc[df['indicators'] == -1], resample_key)
        false_corr_rev_obj = df_set(false_comb_to_rev[resample_key], 
            false_label, True, df.loc[df[false_label + '_rev_indicators'] == 1], 
            resample_key)
        true_corr_obj = df_set(false_corr[resample_key], true_label, False, 
            df.loc[df['indicators'] == 1], resample_key)
        true_corr_rev_obj = df_set(false_comb_to_rev[resample_key], true_label, 
            True,df.loc[df[true_label + '_rev_indicators'] == 1], resample_key)
        # extend dfs list
        dfs.extend([false_corr_obj, false_corr_rev_obj, true_corr_obj, 
            true_corr_rev_obj])

    # for each relevant set
    for df in dfs:
        # plot random / representative correlations 
        plot_corr_sets(graph_bound, df, working_dir, f1type, f2type, var1_names, var2_names, samp_var1, samp_var2)

        # this section plots pvalue and fold pvalue change distributions
        plot_pdist(df, working_dir)


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

    # compute number of infinity and nan entries
    n_infinite = len(fold_pvalues[~np.isfinite(fold_pvalues)])
    n_nan = len(pvalues[np.isnan(pvalues)])

    pvalues = pvalues[~np.isnan(pvalues)]
    fold_pvalues = fold_pvalues[np.isfinite(fold_pvalues)]

    # construct dataframe
    pvalue_df = pd.DataFrame({'pvalues': pvalues, 
        'fold_pvalues': fold_pvalues,
        'log_pvalues': df.rm_subset['logpvals'],
        'log_fold_pvalues': np.log(fold_pvalues)})

    dists = ['pvalues', 'fold_pvalues','log_pvalues','log_fold_pvalues']
    for dist in dists:
        if len(pvalue_df[dist]) > 0:
            title = 'fold_pvalues; n_infinite = ' + str(n_infinite) + ' ' 
                + 'pvalues; n_nan = ' + str(n_nan)
            dist_fp = working_dir + 'graphs/' + df.quadrant + '_' + str(df.k) \
                + '_' + dist + '.png'
            fig = plt.figure()
            sns_plot = sns.distplot(pvalue_df[dist], kde=False, rug=True)
            ax = plt.gca()
            ax.set_title(title, fontsize=10)
            plt.savefig(dist_fp)
            plt.close()

def plot_corr(row, df_folder_fp, f1type, f2type, var1_names, var2_names, 
    samp_var1, samp_var2):
    """
    Helper function for plot_corr_sets(). Plots pairwise correlations within each
    set of correlations as defined by df.
    ----------------------------------------------------------------------------
    INPUTS
    row          - Pandas dataframe row. 
    df_folder_fp - File object. Points to directory where particular set of 
                   plots will be stored.
    f1type       - String. Must be 'map' or 'otu' which specifies parsing 
                   functionality to perform on file 1
    f2type       - String. Same as f1type but for file 2.
    var1_names   - List of strings. List of variables in file 1.
    var2_names   - List of strings. List of variables in file 2.
    samp_var1    - 2D array. Each value in row i col j is the level of 
                   variable j corresponding to sample i in the order that the 
                   samples are presented in samp_ids
    samp_var2    - 2D array. Same as samp_var1 but for file 2.
    """
    var1, var2 = int(row['var1_index']), int(row['var2_index'])

    # obtain variable values
    x = samp_var1[:,var1]
    y = samp_var2[:,var2]
    var1_name = var1_names[var1]
    var2_name = var2_names[var2]

    # if the ftype is OTU, reduce the taxa name into abridged form
    if f1type == 'otu':
        var1_name = parse.read_taxa(var1_name)            
    if f2type == 'otu':
        var2_name = parse.read_taxa(var2_name)

    # consolidate variables into pd dataframe
    pair_df = pd.DataFrame({var1_name:x, var2_name:y})

    # create plot and title
    title = 'p, ext_p = ' + '%.2E' % Decimal(row['pvalues']) + \
            ', ' + '%.2E' % Decimal(row['extreme_p']) + ' ' + \
            'Rsq, ext_r2 = ' + '%.2E' % Decimal(row['r2vals']) + \
            ', ' + '%.2E' % Decimal(row['extreme_r'])
    fig = plt.figure()
    sns_plot = sns.lmplot(var1_name, var2_name, data=pair_df, fit_reg=False)
    ax = plt.gca()
    ax.set_title(title, fontsize=10)
    plt.savefig(df_folder_fp + '/' + str(var1) + '_' + str(var2) + '.png')
    plt.close()

def plot_corr_sets(graph_bound, df, working_dir, f1type, f2type, var1_names, 
    var2_names, samp_var1, samp_var2):
    """  
    Helper function for graph_subsets(). Plots pairwise correlations within each
    set of correlations as defined by df.
    ----------------------------------------------------------------------------
    INPUTS
    graph_bound - Integer. Upper limit of how many graphs to plot in each set.
    df          - Dataframe-sets object as constructed in graph_subsets().
    working_dir - String. Path of working directory specified by user.
    f1type      - String. Must be 'map' or 'otu' which specifies parsing 
                  functionality to perform on file 1
    f2type      - String. Same as f1type but for file 2.
    var1_names  - List of strings. List of variables in file 1.
    var2_names  - List of strings. List of variables in file 2.
    samp_var1   - 2D array. Each value in row i col j is the level of 
                  variable j corresponding to sample i in the order that the 
                  samples are presented in samp_ids
    samp_var2   - 2D array. Same as samp_var1 but for file 2.
    """
    # decide var pairs to plot
    if graph_bound <= len(df.rm_subset):
        # samples without replacement
        df_forplot = df.rm_subset.sample(n=graph_bound, replace=False, 
            weights=None, random_state=42, axis=0)            
    else:
        df_forplot = df.rm_subset
    # name and create folder
    df_folder_fp = working_dir + 'graphs/' + df.quadrant + '_' + str(df.k) + \
        '_' + str(len(df.pairs))
    if df.rev_sign:
        df_folder_fp = df_folder_fp + '_revsign'
    if os.path.exists(df_folder_fp) is not True:
        os.makedirs(df_folder_fp)

    # plot representative plots
    for index, row in df_forplot.iterrows():
        plot_corr(row, df_folder_fp, f1type, f2type, var1_names, 
            var2_names, samp_var1, samp_var2)

###
# Diagnostic plot handling
###

def diag_plots(samp_counter, var1_counter, var2_counter, resample_k, 
    working_dir, paired, samp_var1, samp_var2, n_samp, lof):
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
    lof          - Array. Length n_samp, corresponds to output from 
                   statistics.lof_fit() where i-th entry is -1 if i-th sample is
                   deemed outlier by LOF and 1 otherwise.
    """
    diag_stats = ['samp','var1','var2']
    stats_mapping = {'samp': samp_counter, 
                    'var1': var1_counter,
                    'var2': var2_counter}

    # for each diagnostic quantity
    for stats in diag_stats:
        for i in xrange(resample_k):
            counter = stats_mapping[stats]
            counts = np.zeros(shape=[len(counter[str(i+1)]),2])
            # create 2D array where col 1 = sample/var index and col 2 = 
            # number of times that sample/var appears in CUtIes
            for j in xrange(len(counter[str(i+1)])):
                counts[j] = np.array([j,counter[str(i+1)][j]])

            print_matrix(counts,working_dir + 'data_processing/' + 'counter_' \
                + stats + '_resample' + str(i+1) + '.txt', ['index', 'count'], \
                '\t')

            # create figure, stratifying on lof if quantity is sample
            fig = plt.figure()
            if stats == 'samp':
                counts_df = pd.DataFrame({stats:counts[:,0], 
                    'n_cuties': counts[:,1], 'lof': lof})
                sns_plot = sns.lmplot(stats, 'n_cuties', data=counts_df, 
                    fit_reg=False, hue="lof")
            else: 
                counts_df = pd.DataFrame({stats:counts[:,0], 
                    'n_cuties': counts[:,1]})
                sns_plot = sns.lmplot(stats, 'n_cuties', data=counts_df, 
                    fit_reg=False)

            ax = plt.gca()
            diag_fp = working_dir + 'graphs/' + 'counter_' + stats + \
                '_resample' + str(i+1) + '.png'
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
        f.write('\nThe defaults_fp config file was '+ parse.md5Checksum(defaults_fp))
        f.write('\nThe config_fp config file was '+ parse.md5Checksum(config_fp))        

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


###
# Json matrix handling
###

def create_json_matrix(n_var1, n_var2, n_corr, headers, paired, infln_metrics, 
    FP_infln_sets, initial_sig, TP, point = False):
    """
    Helper function for print_json_matrix. Creates json_matrix for drawing 
    UpSetR plots in R. A json matrix has observations as rows (e.g. each row is 
    a pairwise correlation) and each column is a category. The entry is 1 if a 
    particular sample falls in that category, 0 otherwise.
    ----------------------------------------------------------------------------
    INPUTS
    n_var1        - Integer. Number of var1.
    n_var2        - Integer. Number of var2.
    n_corr        - Integer. Number of correlations being computed. If var1 and 
                    var2 are the same, correlations are double counted but 
                    corr(i,i) are not computed.
    headers       - List of strings. Contains 'var1_index','var2_index' and list 
                    of categories (e.g. CUtIe, Cook's D, DFFITS)
    paired        - Boolean. True if variables are paired (i.e. file 1 and file
                    2 are the same), False otherwise.
    infln_metrics - List of strings. List of categories applied to categorize 
                    each correlation, e.g. 
                    infln_metrics = ['cookd', 'dffits', 'dsr', 'cutie_1sc'] 
    FP_infln_sets - Dictionary. Key is a particular outlier metric in 
                    infln_metrics, entry is a set of tuples classified as 
                    FP by that metric.
    initial_sig   - Set of tuples. Analogous to initial_corr, i.e. set of 
                    correlations initially deemed significant.
    TP            - Integer. Indicator that indicates whether you are creating a
                    matrix to represent overlap between TP (1) or FP (0)
    point         - Boolean. True if each first entry line should be written as 
                    a single number versus as the variable pair. 
    OUTPUTS
    json_matrix   - 2D array. Row is either [x, 1, 0 ... ] or 
                    [var1, var2, 1, 0 ... ] if point is False or True 
                    respectively.
    """
    json_matrix = np.zeros([n_corr, len(headers)])

    row = 0
    for var1 in xrange(n_var1):
        for var2 in xrange(n_var2):
            # the condition ensures that if calculating auto-correlations 
            # i.e when (paired == True) then the matrix will not contain entries
            # where var1 == var2
            if not (paired and (var1 == var2)):
                pair = (var1, var2)
                if point == False:
                    line = [row]
                else:
                    line = [var1, var2]
                for metric in infln_metrics:
                    if pair in initial_sig:
                        if pair in FP_infln_sets[metric]:
                            line.append(1 - TP)
                        else:
                            line.append(TP)
                    else:
                        line.append(0)
                json_matrix[row] = np.array([line])
                row += 1

    return json_matrix


def print_json_matrix(n_var1, n_var2, n_corr, infln_metrics,
    FP_infln_sets, initial_sig, working_dir, paired = False, point = False):
    """
    Creates json_matrix for drawing UpSetR plots in R. A json matrix has 
    observations as rows (e.g. each row is a pairwise correlation) and each 
    column is a category. The entry is 1 if a particular sample falls in that 
    category, 0 otherwise.
    ----------------------------------------------------------------------------
    INPUTS
    n_var1        - Integer. Number of var1.
    n_var2        - Integer. Number of var2.
    n_corr        - Integer. Number of correlations being computed. If var1 and 
                    var2 are the same, correlations are double counted but 
                    corr(i,i) are not computed.
    infln_metrics - List of strings. List of categories applied to categorize 
                    each correlation, e.g. 
                    infln_metrics = ['cookd', 'dffits', 'dsr', 'cutie_1sc'] 
    FP_infln_sets - Dictionary. Key is a particular outlier metric in 
                    infln_metrics, entry is a set of tuples classified as 
                    FP by that metric.
    initial_sig   - Set of tuples. Analogous to initial_corr, i.e. set of 
                    correlations initially deemed significant.
    paired        - Boolean. True if variables are paired (i.e. file 1 and file
                    2 are the same), False otherwise.
    point         - Boolean. True if each first entry line should be written as 
                    a single number versus as the variable pair. 
    """
    # create header row
    if point == False:
        headers = ['corr_row']
    else:
        headers = ['var1_index', 'var2_index']

    for metric in infln_metrics:
        headers.append(metric)

    # create TP matrix
    TP_json_matrix = create_json_matrix(n_var1, n_var2, n_corr, headers, paired, 
        infln_metrics, FP_infln_sets, initial_sig, 1, point)
    print_matrix(TP_json_matrix, 
        working_dir + 'data_processing/TP_json_matrix_' + str(point) + '.txt', 
        headers, ';')
    
    # create FP matrix 
    FP_json_matrix = create_json_matrix(n_var1, n_var2, n_corr, headers, paired, 
        infln_metrics, FP_infln_sets, initial_sig, 0, point)
    print_matrix(FP_json_matrix, 
        working_dir + 'data_processing/FP_json_matrix_' + str(point) + '.txt', 
        headers, ';')

