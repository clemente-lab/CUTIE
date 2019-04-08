#!/usr/bin/env python
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
import math
import itertools
import numpy as np
import minepy
import datetime
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from cutie import parse
from cutie import output
from cutie import utils

def assign_statistics(samp_var1, samp_var2, statistic, pearson_stats,
                      spearman_stats, kendall_stats, mine_stats, mine_bins,
                      pvalue_bins, f1type, log_fp):
    """
    Creates dictionary mapping statistics to 2D matrix containing relevant
    statistics (e.g. pvalue, correlation) for correlation between var i and j.
    Note that because Spearman and MIC do not have meaningful analogs of R2 and
    R respectively, we store an identical value twice.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var1      - 2D array. Each value in row i col j is the level of
                     variable j corresponding to sample i in the order that the
                     samples are presented in samp_ids.
    samp_var2      - 2D array. Same as samp_var1 but for file 2.
    statistic      - String. Describes analysis being performed.
    pearson_stats  - List of strings. Describes possible Pearson-based
                     statistics.
    spearman_stats - List of strings. Describes possible Spearman-based
                     statistics.
    kendall_stats  - List of strings. Describes possible Kendall-based
                     statistics.
    mine_stats     - List of strings. Describes possible MINE-based statistics.
    mine_bins      - 2D Array. Obtained from parse_minep. Each row is in format
                     [MIC_str, pvalue, stderr of pvalue]. Pvalue corresponds to
                     probability of observing MIC_str as or more extreme as
                     observed MIC_str.
    pvalue_bins    - List. Sorted list of pvalues from greatest to least used
                     by MINE to bin the MIC_str.
    f1type         - String. Must be 'map' or 'otu' which specifies parsing
                     functionality to perform on file 1.
    log_fp         - String. File path of log file output.

    OUTPUTS
    stat_to_matrix - Dictionary. Key is string representing particular quantity
                     e.g. pvalue, correlation for given statistic while entry is
                     a 2D array representing numerical value of that quantity.
    """
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    stat_to_matrix = {}

    if statistic in pearson_stats:
        # define function and dictionary mapping string to function
        # for statistic of interest
        functions = ['stats.linregress']
        mapf = {'stats.linregress': stats.linregress}
        # f_stats must match the output of stats.linregress
        f_stats = {'stats.linregress':
                   ['b1', 'b0', 'pcorr', 'ppvalue', 'stderr']}
        stat_dict = initial_stats_SLR(samp_var1, samp_var2, functions, mapf,
                                      f_stats)

        stat_to_matrix['pvalues'] = stat_dict['stats.linregress'][3]
        stat_to_matrix['correlations'] = stat_dict['stats.linregress'][2]
        stat_to_matrix['logpvals'] = np.log(stat_dict['stats.linregress'][3])
        stat_to_matrix['r2vals'] = np.square(stat_dict['stats.linregress'][2])

    elif statistic in spearman_stats:
        functions = ['stats.spearmanr']
        mapf = {'stats.spearmanr': stats.spearmanr}
        f_stats = {'stats.spearmanr': ['scorr', 'spvalue']}
        stat_dict = initial_stats_SLR(samp_var1, samp_var2, functions, mapf,
                                      f_stats)

        stat_to_matrix['pvalues'] = stat_dict['stats.spearmanr'][1]
        stat_to_matrix['logpvals'] = np.log(stat_to_matrix['pvalues'])
        stat_to_matrix['correlations'] = stat_dict['stats.spearmanr'][0]
        stat_to_matrix['r2vals'] = stat_dict['stats.spearmanr'][0]
        # filler, same as correlations

    elif statistic in kendall_stats:
        functions = ['stats.kendalltau']
        mapf = {'stats.kendalltau': stats.kendalltau}
        f_stats = {'stats.kendalltau': ['scorr', 'spvalue']}
        stat_dict = initial_stats_SLR(samp_var1, samp_var2, functions, mapf,
                                      f_stats)

        stat_to_matrix['pvalues'] = stat_dict['stats.kendalltau'][1]
        stat_to_matrix['logpvals'] = np.log(stat_to_matrix['pvalues'])
        stat_to_matrix['correlations'] = stat_dict['stats.kendalltau'][0]
        stat_to_matrix['r2vals'] = stat_dict['stats.kendalltau'][0]
        # filler, same as correlations

    elif statistic in mine_stats:
        MIC_str, MIC_pvalues = initial_stats_MINE(n_var1, samp_var1,
                                                  mine_bins, pvalue_bins)
        stat_to_matrix['pvalues'] = MIC_pvalues
        stat_to_matrix['logpvals'] = np.log(stat_to_matrix['pvalues'])
        stat_to_matrix['correlations'] = MIC_str
        stat_to_matrix['r2vals'] = MIC_str
        # filler, same as correlations

    else:
        output.write_log('Invalid statistic chosen', log_fp)

    return stat_to_matrix


def initial_stats_SLR(samp_var1, samp_var2, functions, mapf, f_stats):
    """
    Helper function for assign_statistics. Forks between SLR (simple linear
    regression/inclusive of Pearson and Spearman) and MINE. Computes an initial
    set of statistics per the specified functions. Returns a dict where the key
    is a statistical function and the element is an initial matrix with
    dimensions n_rel_stats x n_var1 x n_var2, corresponding to the relevant
    statistics from simple linear regression (SLR) between each var1 and var2.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var1  - 2D array. Each value in row i col j is the level of variable j
                 corresponding to sample i in the order that the samples are
                 presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.
    functions  - List of strings. Function names verbatim from libraries.
    mapf       - Dictionary. Maps function names to the function object.
    f_stats    - Dictionary. Maps function name to list of output strings
                 corresponding to what each function returns.

    OUTPUTS
    statistics - Dictionary. Each key is a particular statistics function and
                 each entry is a 3D np array where depth k, row i, col j corresponds
                 to the value of that quantity k for the correlation between
                 var i and var j.
    """
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    stat_dict = {}

    # retrieve relevant stats and create dictionary entry, 3D array
    for f in functions:
        rel_stats = f_stats[f]
        stat_dict[f] = np.zeros((len(rel_stats), n_var1, n_var2))

    # subset the data matrices into the cols needed
    for var1 in range(n_var1):
        for var2 in range(n_var2):
            var1_values = samp_var1[:, var1]
            var2_values = samp_var2[:, var2]
            stacked = np.stack([var1_values, var2_values], 0)
            # remove NANs
            stacked = stacked[:, np.all(~np.isnan(stacked), axis=0)]
            var1_values = stacked[0]
            var2_values = stacked[1]
            for f in functions:
                # values is a list of the relevant_stats in order
                if len(var1_values) == 0 or len(var2_values) == 0:
                    values = np.zeros([len(f_stats[f])])
                    values[:] = np.nan
                else:
                    values = mapf[f](var1_values, var2_values)
                for s in range(len(values)):
                    stat_dict[f][s][var1][var2] = values[s]

    return stat_dict


def initial_stats_MINE(n_var, samp_var, mine_bins, pvalue_bins):
    """
    Helper function for assign_statistics.  Computes an initial
    set of values (MIC_str and corresponding pvalue) for the MIC statistic for
    each correlation. Note currently the MINE API only enables calculation for
    auto-correlations within a single dataframe i.e. paired must be True.
    ----------------------------------------------------------------------------
    INPUTS
    n_var       - Integer. Number of variables.
    samp_var    - 2D array. Each value in row i col j is the level of variable j
                  corresponding to sample i in the order that the samples are
                  presented in samp_ids.
    mine_bins   - 2D Array. Obtained from parse_minep. Each row is in format
                  [MIC_str, pvalue, stderr of pvalue]. Pvalue corresponds to
                  probability of observing MIC_str as or more extreme as
                  observed MIC_str.
    pvalue_bins - List. Sorted list of pvalues from greatest to least used by
                  MINE to bin the MIC_str.

    OUTPUTS
    MIC_str     - 2D array. Entry in i-th row, j-th column corresponds to MIC
                  strength between var i and var j.
    MIC_pvalues - 2D array. Entry in i-th row, j-th column corresponds to pvalue
                  of MIC str between var i and var j.
    """
    df = pd.DataFrame(samp_var)

    # mine accepts files in OTU table format, so need to transpose
    df = df.T

    # drop NA
    # df = df.dropna(how = 'any', axis = 1)
    MIC_flat = minepy.pstats(df, alpha=0.6, c=15, est="mic_approx")[0]

    # see MINE API docstrings for indexing
    # http://minepy.readthedocs.io/en/latest/python.html
    MIC_str = np.zeros(shape=[n_var, n_var])
    for i in range(n_var):
        for j in range(n_var):
            if i == j:
                MIC_str[i][j] = 1
            elif i < j:
                k = int(abs(n_var*i - i*(i+1)/2 - i - 1 + j))
                MIC_str[i][j] = MIC_flat[k]
                MIC_str[j][i] = MIC_flat[k]

    # convert MIC strength into p value
    MIC_pvalues = np.ones(shape=[n_var, n_var])
    for i in range(n_var):
        for j in range(n_var):
            MIC_pvalues[i][j] = str_to_pvalues(pvalue_bins, MIC_str[i][j],
                                               mine_bins)

    return MIC_str, MIC_pvalues

def set_threshold(pvalues, alpha, mc, log_fp, paired=False):
    """
    Computes p-value threshold for alpha according to FDR, Bonferroni, or FWER.
    ----------------------------------------------------------------------------
    INPUTS
    pvalues   - 2D array. Entry row i, col j represents p value of correlation
                between i-th var1 and j-th var2.
    alpha     - Float. Original cut-off for alpha (0.05).
    mc        - String. Form of multiple corrections to use (nomc: none, bc:
                bonferroni, fwer: family-wise error rate, fdr: false discovery
                rate).
    paired    - Boolean. True if variables are paired (i.e. file 1 and file
                2 are the same), False otherwise.

    OUTPUTS
    threshold - Float. Cutoff of pvalues.
    """
    output.write_log('The type of mc correction used was ' + mc, log_fp)
    pvalues_copy = np.copy(pvalues)
    if paired == True:
        # fill the upper diagonal with nan as to not double count pvalues in FDR
        pvalues_copy[np.triu_indices(pvalues_copy.shape[1], 0)] = np.nan
        # currently computing all pairs double counting
        n_corr = np.size(pvalues_copy, 1) * (np.size(pvalues_copy, 1) - 1)
    else:
        n_corr = np.size(pvalues_copy, 0) * np.size(pvalues_copy, 1)

    # determine threshold based on multiple comparisons setting
    pvalues_copy = np.sort(pvalues_copy.flatten())
    pvalues_copy = pvalues_copy[~np.isnan(pvalues_copy)]
    if mc == 'nomc':
        threshold = alpha
    elif mc == 'bc':
        threshold = alpha / pvalues_copy.size
    elif mc == 'fwer':
        threshold = 1.0 - (1.0 - alpha) ** (1/(pvalues_copy.size))
    elif mc == 'fdr':
        # compute FDR cutoff
        # https://brainder.org/2011/09/05/fdr-corrected-fdr-adjusted-p-values/
        # http://www.biostathandbook.com/multiplecomparisons.html
        cn = 1.0
        thresholds = np.array([(float(k+1))/(len(pvalues_copy))
                               * alpha / cn for k in range(len(pvalues_copy))])
        compare = np.where(pvalues_copy <= thresholds)[0]
        if len(compare) is 0:
            threshold = alpha
            output.write_log('Warning: no p-values below threshold, defaulted \
                with min(p) = ' + str(min(pvalues_copy)), log_fp)
        else:
            threshold = thresholds[max(compare)]
    output.write_log('The threshold value was ' + str(threshold), log_fp)
    return threshold, n_corr



###
# Zero handling for log transform
###

def multi_zeros(n_samp, n_var, samp_var):
    """
    INPUTS
    samp_var: 2D array where each entry in row i col j refers to relative
                      abundance of var1 j in sample i

    OUTPUTS
    samp_var_mr:     2D corrected matrix (0's replaced with threshold)
    samp_var_clr:    2D centered log ratio matrix, each row of mr divided by its geometric mean
    samp_var_lclr:   2D log of CLR matrix, log of each row
    samp_var_varlog: 1D variance of lclr matrix, element j refers to variance of col j
    correction:       threshold used for correction (currently min(samp_var_matrix / 2))
    n_zero:           number of 0's detected in the original samp_var_matrix

    FUNCTION
    Eliminates 0's from a matrix and replaces it with a multiplicative threshold
    correction, using the smallest value divided by 2 as the replacement.
    """
    # create working copy
    samp_var_mr = np.copy(samp_var)

    # obtain 0 correction value
    correction = zero_replacement(samp_var)

    # replace 0s with correction
    samp_var_mr = multi_replacement(correction, samp_var, samp_var_mr)

    # create array of geometric means for log clr correction
    samp_var_gm = np.zeros(n_samp)
    for i in range(n_samp):
        samp_var_gm[i] = math.exp(sum(np.log(samp_var_mr[i])) / float(n_var))

    # create log clr correction
    samp_var_clr = samp_var_mr / samp_var_gm[:, None]
    samp_var_lclr = np.log(samp_var_clr)

    # create array of variances
    samp_var_varlog = np.zeros(n_var)
    for i in range(n_var):
        samp_var_varlog[i] = np.var(samp_var_lclr[:, i])

    return samp_var_mr, samp_var_clr, samp_var_lclr, samp_var_varlog

def multi_replacement(correction, samp_var, samp_var_mr):
    """
    Helper function for multi_zeros(). Replaces 0 values in a 2D array with a
    value specified by correction in accordance with the multiplicative
    replacement procedure for dealing with 0s.
    ----------------------------------------------------------------------------
    INPUTS
    correction  - Float. Value with which to replace 0s.
    samp_var    - 2D array. Each value in row i col j is the level of variable j
                  corresponding to sample i in the order that the samples are
                  presented in samp_ids.
    samp_var_mr - 2D array. Copy of samp_var in which 0s will be replaced.
    """
    n_var, n_var, n_samp = utils.get_param(samp_var, samp_var)

    samp_var_mr[samp_var_mr == 0] = correction

    # correct non-zero values
    for i in range(n_samp):
        nrow_zero = len(np.where(samp_var[i] == 0)[0])
        for j in range(n_var):
            if samp_var[i][j] != 0:
                samp_var_mr[i][j] = samp_var_mr[i][j] * (1 - nrow_zero * correction)

    return samp_var_mr

def zero_replacement(samp_var):
    """
    Helper function for multi_zeros(). Obtains value for zero replacement by
    taking the minimum value observed in samp_var and replacing it by its square or
    its half if it is greater than one. divided by 2.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var - 2D array. Each value in row i col j is the level of variable j
               corresponding to sample i in the order that the samples are
               presented in samp_ids.
    """
    # find min non-zero value
    min_value = min(samp_var[np.nonzero(samp_var)])
    if min_value < 1:
        correction = min_value ** 2 # or use divided by 2)
    else:
        correction = min_value / 2

    return correction

###
# Pointwise diagnostics
###

def resample1_cutie_pc(var1_index, var2_index, samp_var1, samp_var2, influence1,
                       influence2, threshold, sign, fold, fold_value):
    """
    Takes a given var1 and var2 by indices and recomputes Pearson correlation
    by removing 1 out of n (sample_size) points from samp_ids.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index - Integer. Index for variable from file 1 in pairwise correlation.
    var2_index - Integer. Index for variable from file 2 in pairwise correlation.
    samp_var1  - 2D array. Each value in row i col j is the level of variable j
                 corresponding to sample i in the order that the samples are
                 presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.
    influence1 - sm.OLS object. Not relevant to Pearson/Spearman but needed as a
                 placeholder argument (for Cook's D, etc.)
    influence2 - sm.OLS object. Same remarks as for influence1.
    threshold  - Float. Level of significance testing (after adjusting for
                 multiple comparisons).
    sign       - Integer. -1 or 1, depending on original sign of correlation to
                 check against following re-evaluation.
    fold       - Boolean. Determines whether you require the new P value to be a
                 certain fold greater to be classified as a CUtIe.
    fold_value - Float. Determines fold difference constraint imposed on the
                 resampled p-value needed for a correlation to be classified as
                 a CUtIe.

    OUTPUTS
    reverse    - 1D array. Index i is 1 if the correlation changes sign upon
                 removing sample i.
    exceeds    - 1D array. Index i is 1 if removing that sample causes the
                 correlation to become insignificant in at least 1 different
                 pairwise correlations.
    corrs      - 1D array. Contains values of correlation strength with sample i
                 removed.
    p_values   - 1D array. Contains values of pvalues with sample i removed.
    """
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    exceeds, reverse, maxp, minr, var1, var2 = utils.init_var_indicators(var1_index,
                                                                   var2_index,
                                                                   samp_var1,
                                                                   samp_var2,
                                                                   True)
    corrs = np.zeros(n_samp)
    p_values = np.zeros(n_samp)

    # iteratively delete one sample and recompute statistics
    s, i, r, original_p, s = stats.linregress(var1, var2)

    for s in range(n_samp):
        new_var1 = var1[~np.in1d(range(n_samp), s)]
        new_var2 = var2[~np.in1d(range(n_samp), s)]

        # remove NaNs
        new_var1, new_var2 = utils.remove_nans(new_var1, new_var2)

        # compute new p_value and r_value
        p_value, r_value = compute_pc(new_var1, new_var2)


        # update reverse, maxp, and minr
        reverse, maxp, minr = update_rev_extrema_rp(sign, r_value, p_value,
                                                    [s], reverse, maxp, minr,
                                                    True)

        if fold:
            if (p_value > threshold and \
                p_value > pvalues[var1_index][var2_index] * fold_value) or \
                np.isnan(p_value):
                exceeds[s] += 1
        elif p_value > threshold or np.isnan(p_value):
            exceeds[s] += 1

        corrs[s] = r_value
        p_values[s] = p_value

    return reverse, exceeds, corrs, p_values

def resample1_cutie_sc(var1_index, var2_index, samp_var1, samp_var2, influence1,
                       influence2, threshold, sign, fold, fold_value):
    """
    Takes a given var1 and var2 by indices and recomputes Spearman correlation
    by removing 1 out of n (sample_size) points from samp_ids.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index - Integer. Index for variable from file 1 in pairwise correlation.
    var2_index - Integer. Index for variable from file 2 in pairwise correlation.
    samp_var1  - 2D array. Each value in row i col j is the level of variable j
                 corresponding to sample i in the order that the samples are
                 presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.
    influence1 - sm.OLS object. Not relevant to Pearson/Spearman but needed as a
                 placeholder argument (for Cook's D, etc.)
    influence2 - sm.OLS object. Same remarks as for influence1.
    threshold  - Float. Level of significance testing (after adjusting for
                 multiple comparisons)
    sign       - Integer. -1 or 1, depending on original sign of correlation to
                 check against following re-evaluation.
    fold       - Boolean. Determines whether you require the new P value to be a
                 certain fold greater to be classified as a CUtIe.
    fold_value - Float. Determines fold difference constraint imposed on the
                 resampled p-value needed for a correlation to be classified as
                 a CUtIe.

    OUTPUTS
    reverse    - 1D array. Index i is 1 if the correlation changes sign upon
                 removing sample i.
    exceeds    - 1D array. Index i is 1 if removing that sample causes the
                 correlation to become insignificant in at least 1 different
                 pairwise correlations
    corrs      - 1D array. Contains values of correlation strength with sample i
                 removed.
    p_values   - 1D array. Contains values of pvalues with sample i removed.
    """
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    exceeds = np.zeros(n_samp)
    reverse = np.zeros(n_samp)
    corrs = np.zeros(n_samp)
    p_values = np.zeros(n_samp)
    var1_values = samp_var1[:, var1_index]
    var2_values = samp_var2[:, var2_index]

    # iteratively delete one sample and recompute statistics
    for sample_index in range(n_samp):
        new_var1_values = var1_values[~np.in1d(range(n_samp), sample_index)]
        new_var2_values = var2_values[~np.in1d(range(n_samp), sample_index)]
        if new_var1_values.size <= 3 or new_var2_values.size <= 3:
            p_value = 1
            corr = 0
        else:
            corr, p_value = stats.spearmanr(new_var1_values, new_var2_values)
        if fold:
            if (p_value > threshold and p_value > original_p * fold_value) or \
                np.isnan(p_value):
                exceeds[sample_index] = 1
        elif p_value > threshold or np.isnan(p_value):
            exceeds[sample_index] = 1
        if np.sign(corr) != sign:
            reverse[sample_index] = 1
        corrs[sample_index] = corr
        p_values[sample_index] = p_value

    return reverse, exceeds, corrs, p_values

def cookd(var1_index, var2_index, samp_var1, samp_var2,
          influence1, influence2, threshold, sign, fold, fold_value):
    """
    Takes a given var1 and var2 by indices and recomputes Cook's D for each i-th
    sample.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index - Integer. Index for variable from file 1 in pairwise correlation.
    var2_index - Integer. Index for variable from file 2 in pairwise correlation.
    samp_var1  - 2D array. Each value in row i col j is the level of variable j
                 corresponding to sample i in the order that the samples are
                 presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.
    influence1 - sm.OLS object. Not relevant to Pearson/Spearman but needed as a
                 placeholder argument (for Cook's D, etc.)
    influence2 - sm.OLS object. Same remarks as for influence1.
    threshold  - Float. Level of significance testing (after adjusting for
                 multiple comparisons)
    sign       - Integer. -1 or 1, depending on original sign of correlation to
                 check against following re-evaluation.
    fold       - Boolean. Determines whether you require the new P value to be a
                 certain fold greater to be classified as a CUtIe.
    fold_value - Float. Determines fold difference constraint imposed on the
                 resampled p-value needed for a correlation to be classified as
                 a CUtIe.

    OUTPUTS
    reverse    - 1D array. Index i is 1 if the correlation changes sign upon
                 removing sample i.
    exceeds    - 1D array. Index i is 1 if removing that sample causes the
                 correlation to become insignificant in at least 1 different
                 pairwise correlations
    corrs      - 1D array. Contains values of correlation strength with sample i
                 removed.
    p_values   - 1D array. Contains values of pvalues with sample i removed.
    """
    n_samp = np.size(samp_var1, 0)
    # reverse is 0 because sign never changes
    reverse = np.zeros(n_samp)
    exceeds = np.zeros(n_samp)
    #c is the distance and p is p-value
    (c, p) = influence1.cooks_distance
    for i in range(len(c)):
        if c[i] > 1 or np.isnan(c[i]) or c[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds, c, p


def dffits(var1_index, var2_index, samp_var1, samp_var2,
           influence1, influence2, threshold, sign, fold, fold_value):
    """
    Takes a given var1 and var2 by indices and recomputes DFFITS for each i-th
    sample.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index - Integer. Index for variable from file 1 in pairwise correlation.
    var2_index - Integer. Index for variable from file 2 in pairwise correlation.
    samp_var1  - 2D array. Each value in row i col j is the level of variable j
                 corresponding to sample i in the order that the samples are
                 presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.
    influence1 - sm.OLS object. Not relevant to Pearson/Spearman but needed as a
                 placeholder argument (for Cook's D, etc.)
    influence2 - sm.OLS object. Same remarks as for influence1.
    threshold  - Float. Level of significance testing (after adjusting for
                 multiple comparisons)
    sign       - Integer. -1 or 1, depending on original sign of correlation to
                 check against following re-evaluation.
    fold       - Boolean. Determines whether you require the new P value to be a
                 certain fold greater to be classified as a CUtIe.
    fold_value - Float. Determines fold difference constraint imposed on the
                 resampled p-value needed for a correlation to be classified as
                 a CUtIe.

    OUTPUTS
    reverse    - 1D array. Index i is 1 if the correlation changes sign upon
                 removing sample i.
    exceeds    - 1D array. Index i is 1 if removing that sample causes the
                 correlation to become insignificant in at least 1 different
                 pairwise correlations
    dffits_    - 1D array of value of dffits strength with index i removed
    placehold  - List. Placeholder representing threshold used.
    """
    n_samp = np.size(samp_var1, 0)
    reverse = np.zeros(n_samp)
    exceeds = np.zeros(n_samp)
    dffits_, dffits_threshold = influence1.dffits
    for i in range(n_samp):
        if dffits_[i] > dffits_threshold or dffits_[i] < -dffits_threshold or \
        np.isnan(dffits_[i]) or dffits_[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds, dffits_, [dffits_threshold] * n_samp


def dsr(var1_index, var2_index, samp_var1, samp_var2,
        influence1, influence2, threshold, sign, fold, fold_value):
    """
    Takes a given var1 and var2 by indices and recomputes DFFITS for each i-th
    sample.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index - Integer. Index for variable from file 1 in pairwise correlation.
    var2_index - Integer. Index for variable from file 2 in pairwise correlation.
    samp_var1  - 2D array. Each value in row i col j is the level of variable j
                 corresponding to sample i in the order that the samples are
                 presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.
    influence1 - sm.OLS object. Not relevant to Pearson/Spearman but needed as a
                 placeholder argument (for Cook's D, etc.)
    influence2 - sm.OLS object. Same remarks as for influence1.
    threshold  - Float. Level of significance testing (after adjusting for
                 multiple comparisons)
    sign       - Integer. -1 or 1, depending on original sign of correlation to
                 check against following re-evaluation.
    fold       - Boolean. Determines whether you require the new P value to be a
                 certain fold greater to be classified as a CUtIe.
    fold_value - Float. Determines fold difference constraint imposed on the
                 resampled p-value needed for a correlation to be classified as
                 a CUtIe.

    OUTPUTS
    reverse    - 1D array. Index i is 1 if the correlation changes sign upon
                 removing sample i.
    exceeds    - 1D array. Index i is 1 if removing that sample causes the
                 correlation to become insignificant in at least 1 different
                 pairwise correlations
    dsr_       - 1D array of value of dffits strength with index i removed
    dsr_       - 1D array. Repeated as placeholder.

    """
    n_samp = np.size(samp_var1, 0)
    reverse = np.zeros(n_samp)
    exceeds = np.zeros(n_samp)
    dsr_ = influence1.resid_studentized_external
    for i in range(n_samp):
        if dsr_[i] < -2 or dsr_[i] > 2 or np.isnan(dsr_[i]) or dsr_[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds, dsr_, dsr_


def return_influence(var1, var2, samp_var1, samp_var2):
    """
    Compute and return influence objects holding regression diagnostics
    ----------------------------------------------------------------------------
    INPUTS
    var1       - Integer. Index of variable in file 1.
    var2       - Integer. Index of variable in file 2.
    samp_var1  - 2D array. Each value in row i col j is the level of
                 variable j corresponding to sample i in the order that the
                 samples are presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.

    OUTPUTS
    influence1 - sm.OLS object. Not relevant to Pearson/Spearman but needed as a
                 placeholder argument (for Cook's D, etc.)
    influence2 - sm.OLS object. Same remarks as for influence1.

    """
    x_old = samp_var1[:, var1]
    y_old = samp_var2[:, var2]
    # add constant for constant term in regression
    x = sm.add_constant(x_old)
    y = sm.add_constant(y_old)
    # compute models with x and y as independent vars, respectively
    model1 = sm.OLS(y_old, x)
    fitted1 = model1.fit()
    influence1 = fitted1.get_influence()
    model2 = sm.OLS(x_old, y)
    fitted2 = model2.fit()
    influence2 = fitted2.get_influence()
    return influence1, influence2

def calculate_FP_sets(initial_corr, corrs, samp_var1, samp_var2, infln_metrics,
                      infln_mapping, threshold, fold, fold_value):
    """
    Determine which correlations (variable pairs) belong in which
    infln_metric_FP sets.
    ----------------------------------------------------------------------------
    INPUTS
    initial_corr  - Set of integer tuples. Contains variable pairs initially
                    classified as significant (forward CUtIe) or insignificant
                    (reverse CUtIe). Note variable pairs (i,j) and (j,i) are
                    double counted.
    infln_metrics - List. Contains strings of infln_metrics (such as 'cookd').
    infln_mapping - Dictionary. Maps strings of function names to function
                    objects (e.g. 'cookd')
    corrs         - 2D array. Contains values of correlation strength between
                    var i and var j.
    samp_var1     - 2D array. Each value in row i col j is the level of
                    variable j corresponding to sample i in the order that the
                    samples are presented in samp_ids.
    samp_var2     - 2D array. Same as samp_var1 but for file 2.
    threshold     - Float. Level of significance testing (after adjusting for
                    multiple comparisons)
    fold          - Boolean. Determines whether you require the new P value to
                    be a certain fold greater to be classified as a CUtIe.
    fold_value    - Float. Determines fold difference constraint imposed on the
                    resampled p-value needed for a correlation to be classified as
                    a CUtIe.

    OUTPUTS
    FP_infln_sets - Dictionary. Key is particular outlier metric, entry is a set
                    of variable pairs classified as FP according to that metric.
    """
    FP_infln_sets = {}

    # initialize dict
    for metric in infln_metrics:
        FP_infln_sets[metric] = set()

    # determine if each initial_corr correlation belongs in each metric FP set
    for pair in initial_corr:
        var1, var2 = pair
        influence1, influence2 = return_influence(var1, var2, samp_var1,
                                                  samp_var2)
        for metric in infln_metrics:
            sign = np.sign(corrs[var1][var2])
            reverse, exceeds, corr_values, thresholds = infln_mapping[metric](
                var1, var2, samp_var1, samp_var2, influence1, influence2,
                threshold, sign, fold, fold_value)

            # if exceeds == 0 then it is a TP
            if exceeds.sum() != 0:
                FP_infln_sets[metric].add(pair)

    return FP_infln_sets


def pointwise_comparison(samp_var1, samp_var2, pvalues, corrs, working_dir,
                         n_corr, initial_corr, threshold, statistic, fold_value,
                         log_fp, paired, fold):
    """
    Perform pointwise analysis of each correlation, comparing between CUtIe,
    Cook's D, DFFITS (and optionally DSR). Logs number of correlations belonging
    to each set (Venn Diagram) of outlier metrics as well as a JSON table.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var1    - 2D array. Each value in row i col j is the level of
                   variable j corresponding to sample i in the order that the
                   samples are presented in samp_ids.
    samp_var2    - 2D array. Same as samp_var1 but for file 2.
    pvalues      - 2D array. Contains pvalue between var i and var j.
    corrs        - 2D array. Contains values of correlation strength between
                   var i and var j.
    working_dir  - String. File path of working directory specified by user.
                   Should end in '/'
    n_corr       - Integer. Number of correlations being computed. If var1 and
                   var2 are the same, correlations are double counted but
                   corr(i,i) are not computed.
    initial_corr - Set of integer tuples. Contains variable pairs initially
                   classified as significant (forward CUtIe) or insignificant
                   (reverse CUtIe). Note variable pairs (i,j) and (j,i) are
                   double counted.
    threshold    - Float. Level of significance testing (after adjusting for
                   multiple comparisons)
    statistic    - String. Describes analysis being performed.
    fold_value   - Float. Determines fold difference constraint imposed on the
                   resampled p-value needed for a correlation to be classified
                   as a CUtIe.
    log_fp       - String. Points to file path of log file.
    fold         - Boolean. Determines whether you require the new P value to
                   be a certain fold greater to be classified as a CUtIe.
    """
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    infln_metrics = ['cookd', 'cutie_1pc'] #, 'dffits', 'dsr'] # 'dsr'
    infln_mapping = {
        'cutie_1pc': resample1_cutie_pc,
        'cookd': cookd,
        #'dffits': dffits,
        #'dsr': dsr
        }

    # key is metric, entry is set of points FP to that metric
    FP_infln_sets = calculate_FP_sets(initial_corr, corrs, samp_var1, samp_var2,
                                      infln_metrics, infln_mapping, threshold,
                                      fold, fold_value)

    # create list of sets
    FP_infln_sets_list = []
    for metric in infln_metrics:
        FP_infln_sets_list.append(FP_infln_sets[metric])

    region_combs, region_sets = utils.calculate_intersection(infln_metrics,
                                                       FP_infln_sets_list,
                                                       log_fp)

    # base regions == infln_metriics for this
    output.generate_pair_matrix(infln_metrics, FP_infln_sets, n_var1, n_var2,
                         samp_var1, samp_var2, infln_metrics, working_dir)

    # report results
    for metric in infln_metrics:
        metric_FP = FP_infln_sets[metric]
        output.write_log('The number of false correlations according to ' + \
            metric + ' is ' + str(len(metric_FP)), log_fp)
        output.write_log('The number of true correlations according to ' + \
            metric + ' is ' + str(len(initial_corr) - len(metric_FP)), log_fp)

    return infln_metrics, infln_mapping, FP_infln_sets, region_combs, region_sets


def log_transform(samp_var, working_dir, var_number):
    """
    Computes log-transform of 2D np array after 0 replacement is performed.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var     - 2D array. Each value in row i col j is the level of
                   variable j corresponding to sample i in the order that the
                   samples are presented in samp_ids.
    working_dir  - String. File path of working directory specified by user.
                   Should end in '/'
    var_number   - Integer. Represents if file 1 or file 2 is being
                   log-transformed (or both).
    """
    n_var, n_var, n_samp = utils.get_param(samp_var, samp_var)

    samp_var_mr, samp_var_clr, samp_var_lclr, samp_var_varlog = \
        multi_zeros(n_samp, n_var, samp_var)

    header = [str(x+1) for x in range(n_var)]
    output.print_matrix(samp_var_mr, working_dir + 'data_processing/samp_var' + \
        str(var_number) + '_mr.txt', header, '\t')

    return np.log(samp_var_mr)


def get_initial_corr(n_var1, n_var2, pvalues, threshold, paired):
    """
    Determine list of initially candidate correlations for screening (either sig
    or nonsig, if performing CUtIe or reverse-CUtIe respectively).
    ----------------------------------------------------------------------------
    INPUTS
    n_var1       - Integer. Number of variables in file 1.
    n_var2       - Integer. Number of variables in file 2.
    pvalues      - 2D array. Contains pvalue between var i and var j.
    threshold    - Float. Level of significance testing (after adjusting for
                   multiple comparisons)

    OUTPUTS
    initial_corr - Set of tuples. Variable pairs for screening.
    all_pairs    - Set of tuples. All variable pairs are accounted for,
                   with corr(i,j) double counted but corr(i,i) omitted if paired
                   is true.
    """
    initial_corr = []
    all_pairs = []
    for var1 in range(n_var1):
        for var2 in range(n_var2):
            pair = (var1, var2)
            # if paired is true and var1 == var2,
            # then the statement is overall false
            if not (paired and (var1 == var2)):
                all_pairs.append(pair)
            # if variables are paired i.e. both x1 and x2 are the same
            # then don't compute corr(i,i)
            if pvalues[var1][var2] < threshold and \
            not (paired and (var1 == var2)):
                initial_corr.append(pair)

    return initial_corr, all_pairs

###
# RESAMPLE K
###

def updatek_cutie(initial_corr, pvalues, samp_var1, samp_var2, threshold,
                  resample_k, corrs, fold, fold_value, working_dir, CI_method,
                  forward_stats, reverse_stats, pvalue_bins, mine_bins, paired,
                  statistic, n_replicates=1000):
    """
    Perform cutie resampling up to k points or other statistical analysis
    (bootstrapping, jackknifing).
    ----------------------------------------------------------------------------
    INPUTS
    initial_corr      - Set of integer tuples. Contains variable pairs initially
                        classified as significant (forward CUtIe) or
                        insignificant (reverse CUtIe). Note variable pairs (i,j)
                        and (j,i) are double counted.
    pvalues           - 2D array. Entry row i, col j represents p value of
                        correlation between i-th var1 and j-th var2.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids when parsed.
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    threshold         - Float. Level of significance testing (after adjusting
                        for multiple comparisons)
    resample_k        - Integer. Number of points being resampled by CUtIe.
    corrs             - 2D array. Contains values of correlation strength between
                        var i and var j.
    fold              - Boolean. Determines whether you require the new P value
                        to be a certain fold greater to be classified as a CUtIe.
    fold_value        - Float. Determines fold difference constraint imposed on
                        the resampled p-value needed for a correlation to be
                        classified as a CUtIe.
    working_dir       - String. File path of working directory specified by user.
    method            - String. 'log', 'cbrt' or 'none' depending on method used
                        for evaluating confidence interval (bootstrapping and
                        jackknifing only).
    forward_stats     - List of strings. Contains list of statistics e.g. 'kpc'
                        'jkp' that pertain to forward (non-reverse) CUtIe
                        analysis.
    reverse_stats     - List of strings. Contains list of statistics e.g. 'rpc'
                        'rjkp' that pertain to reverse CUtIe analysis.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    paired            - Boolean. True if variables are paired (i.e. file 1 and
                        file 2 are the same), False otherwise.
    statistic         - String. Describes analysis being performed.
    n_replicates      - Integer. Number of replicates for bootstrap resampling.

    OUTPUTS
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
    extrema_p         - Dictionary. Key is number of points being resampled and
                        entry is 2D array where row i col j refers to worst or
                        best (if CUtIe or reverse CUtIe is run, respectively)
                        for correlation between var i and var j.
    extrema_r         - Dictionary. Same as extrema_p except values stored are
                        correlation strengths.
    samp_counter      - Dictionary. Key is the index of CUtIe resampling
                        (k = 1, 2, 3, ... etc.) and entry is an array of length
                        n_samp corresponding to how many times the i-th sample
                        appears in CUtIe's when evaluated at resampling = k
                        points).
    var1_counter      - Dictionary. Key is the index of CUtIe resampling
                        (k = 1, 2, 3, ... etc.) and entry is an array of length
                        n_var1 corresponding to how many times the j-th variable
                        appears in CUtIe's when evaluated at resampling = k
                        points)
    var2_counter      - Same as var1_counter except for var2.
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
    """
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    # raise error if resampling too many points
    if resample_k > n_samp - 3:
        raise ValueError('Too many points specified for resampling for size %s'
                         % (str(len(samp_ids))))

    (true_corr, true_comb_to_rev, false_comb_to_rev,
     extrema_p, extrema_r) = utils.initialize_stat_dicts(resample_k, n_var1, n_var2,
                                                   statistic, forward_stats,
                                                   reverse_stats)

    # true sig and false insig are lists of tuples
    # where each tuple is a pair of variable indices

    # TP_comb_to_rev, etc. all represent dicts
    # where the key is the number of points being resampled and the entry is
    # a list of tuples if that particular correlation of variable pair
    # undergoes a sign reversed

    # P_wprst_p and P_worst_r etc. all represent dict
    # where the key is the number of points being resampled and the entry is
    # a 2D array where entry i , j is the worst or best p value or R value

    # separate FP and TP
    (true_corr, true_comb_to_rev, false_comb_to_rev, corr_extrema_p,
     corr_extrema_r, samp_counter, var1_counter, var2_counter, exceeds_points,
     rev_points) = cutiek_true_corr(initial_corr, samp_var1, samp_var2, pvalues,
                                    corrs, threshold, paired, statistic,
                                    forward_stats, reverse_stats, resample_k,
                                    true_corr, true_comb_to_rev,
                                    false_comb_to_rev, extrema_p, extrema_r,
                                    fold, fold_value, n_replicates, CI_method,
                                    pvalue_bins, mine_bins)

    return (true_corr, true_comb_to_rev, false_comb_to_rev, corr_extrema_p,
            corr_extrema_r, samp_counter, var1_counter, var2_counter,
            exceeds_points, rev_points)


def cutiek_true_corr(initial_corr, samp_var1, samp_var2, pvalues, corrs,
                     threshold, paired, statistic, forward_stats, reverse_stats,
                     resample_k, true_corr, true_comb_to_rev, false_comb_to_rev,
                     corr_extrema_p, corr_extrema_r, fold, fold_value,
                     n_replicates, CI_method, pvalue_bins=0, mine_bins=0):
    """
    Helper function for updatek_cutie(). Determine true correlations via
    resampling of k points. Defaults for pvalue_bins and mine_bins are to handle
    cases where statistic is not MINE based.
    ----------------------------------------------------------------------------
    INPUTS
    initial_corr      - Set of integer tuples. Contains variable pairs initially
                        classified as significant (forward CUtIe) or
                        insignificant (reverse CUtIe). Note variable pairs (i,j)
                        and (j,i) are double counted.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids when parsed.
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    pvalues           - 2D array. Entry row i, col j represents p value of
                        correlation between i-th var1 and j-th var2.
    corrs             - 2D array. Contains values of correlation strength
                        between var i and var j.
    threshold         - Float. Level of significance testing (after adjusting
                        for multiple comparisons)
    paired            - Boolean. True if variables are paired (i.e. file 1 and
                        file 2 are the same), False otherwise.
    statistic         - String. Describes analysis being performed.
    forward_stats     - List of strings. Contains list of statistics e.g. 'kpc'
                        'jkp' that pertain to forward (non-reverse) CUtIe
                        analysis.
    reverse_stats     - List of strings. Contains list of statistics e.g. 'rpc'
                        'rjkp' that pertain to reverse CUtIe analysis.
    resample_k        - Integer. Number of points being resampled by CUtIe.
    true_corr         - Set of integer tuples. Contains variable pairs
                        classified as true correlations (TP or FN, depending on
                        forward or reverse CUtIe respectively).
    true_comb_to_rev  - Dictionary. Key is string of number of points being
                        resampled, and entry is a 2D array of indicators where
                        the entry in the i-th row and j-th column is 1 if that
                        particular correlation in the set of true_corr (either
                        TP or FN) reverses sign upon removal of a point.
    false_comb_to_rev - Dictionary. Same as true_comb_to_rev but for TN/FP.
    corr_extrema_p    - Dictionary. Key is number of points being resampled and
                        entry is 2D array where row i col j refers to worst or
                        best (if CUtIe or reverse CUtIe is run, respectively)
                        for correlation between var i and var j.
    corr_extrema_r    - Dictionary. Same as extrema_p except values stored are
                        correlation strengths.
    fold              - Boolean. Determines whether you require the new P value
                        to be a certain fold greater to be classified as a CUtIe.
    fold_value        - Float. Determines fold difference constraint imposed on
                        the resampled p-value needed for a correlation to be
                        classified as a CUtIe.
    n_replicates      - Integer. Number of replicates for bootstrap resampling.
    CI_method         - String. 'log', 'cbrt' or 'none' depending on method used
                        for evaluating confidence interval (bootstrapping and
                        jackknifing only).
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.

    OUTPUTS (in addition to the above)
    samp_counter      - Dictionary. Key is the index of CUtIe resampling
                        (k = 1, 2, 3, ... etc.) and entry is an array of length
                        n_samp corresponding to how many times the i-th sample
                        appears in CUtIe's when evaluated at resampling = k
                        points).
    var1_counter      - Dictionary. Key is the index of CUtIe resampling
                        (k = 1, 2, 3, ... etc.) and entry is an array of length
                        n_var1 corresponding to how many times the j-th variable
                        appears in CUtIe's when evaluated at resampling = k
                        points)
    var2_counter      - Same as var1_counter except for var2.
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
    """
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    # determine forward or reverse handling
    if statistic in forward_stats:
        forward = True
    elif statistic in reverse_stats:
        forward = False

    # diagnostic statistics and rev/exceeds trackers
    samp_counter = {}
    var1_counter = {}
    var2_counter = {}
    rev_points = {}
    exceeds_points = {}

    # initialize counter dictionaries for tracking sample and var freq in CUtIes
    for i in range(resample_k):
        samp_counter[str(i+1)] = np.zeros(n_samp)
        var1_counter[str(i+1)] = np.zeros(np.size(samp_var1, 1))
        var2_counter[str(i+1)] = np.zeros(np.size(samp_var2, 1))
        rev_points[str(i+1)] = {}
        exceeds_points[str(i+1)] = {}

    for pair in initial_corr:
        var1, var2 = pair
        # obtain sign of correlation
        sign = np.sign(corrs[var1][var2])
        # indicators for whether correlation is true or not
        truths = np.zeros(n_samp)
        # indicators for whether a sign reverses
        rev_corr = np.zeros(n_samp)

        # resample_k = number of points being resampled
        for i in range(resample_k):
            # corrs is MINE_str
            new_rev_corr, new_truths, extrema_p, extrema_r = evaluate_correlation_k(
                var1, var2, n_samp, samp_var1, samp_var2, pvalues, threshold,
                statistic, i, sign, fold, fold_value, n_replicates, CI_method,
                forward, forward_stats, reverse_stats, pvalue_bins, corrs,
                mine_bins)

            # update the insig-indicators for the k-th resample iteration
            truths = np.add(truths, new_truths)
            rev_corr = np.add(rev_corr, new_rev_corr)

            # update the correlation within the dictionary of worst p and r values
            corr_extrema_p[str(i+1)][var1][var2] = extrema_p
            corr_extrema_r[str(i+1)][var1][var2] = extrema_r

            # if no points cause p value to rise above threshold, insig sums to 0
            # non-CUtIe's follow this path
            if truths.sum() == 0:
                true_corr[str(i+1)].append(pair)
                # if there exists a point where the sign changes
                if rev_corr.sum() != 0:
                    true_comb_to_rev[str(i+1)].append(pair)

            # CUtIe's follow this path
            else:
                # if there exists a point where sign changes
                if rev_corr.sum() != 0:
                    false_comb_to_rev[str(i+1)].append(pair)

                # update outlier counters
                samp_counter[str(i+1)] = np.add(samp_counter[str(i+1)], truths)
                var1_counter[str(i+1)][var1] += 1
                var2_counter[str(i+1)][var2] += 1

            exceeds_points[str(i+1)][str(pair)] = truths
            rev_points[str(i+1)][str(pair)] = rev_corr

    return (true_corr, true_comb_to_rev, false_comb_to_rev, corr_extrema_p,
            corr_extrema_r, samp_counter, var1_counter, var2_counter,
            exceeds_points, rev_points)

def evaluate_correlation_k(var1, var2, n_samp, samp_var1, samp_var2, pvalues,
                           threshold, statistic, index, sign, fold, fold_value,
                           n_replicates, CI_method, forward, forward_stats,
                           reverse_stats, pvalue_bins, mine_str, mine_bins):
    """
    Helper function for cutiek_true_corr(). Evaluates a given var1, var2
    correlation at the resample_k = i level.
    ----------------------------------------------------------------------------
    INPUTS
    var1              - Integer. Index of variable in file 1.
    var2              - Integer. Index of variable in file 2.
    new_samp          - Integer. Number of samples.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids when parsed.
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    pvalues           - 2D array. Entry row i, col j represents p value of
                        correlation between i-th var1 and j-th var2.
    threshold         - Float. Level of significance testing (after adjusting
                        for multiple comparisons)
    statistic         - String. Describes analysis being performed.
    index             - Integer. Number of points being resampled.
    sign              - Integer. -1 or 1, depending on original sign of
                        correlation to check against following re-evaluation.
    fold              - Boolean. Determines whether you require the new P value
                        to be a certain fold greater to be classified as a CUtIe.
    fold_value        - Float. Determines fold difference constraint imposed on
                        the resampled p-value needed for a correlation to be
                        classified as a CUtIe.
    n_replicates      - Integer. Number of replicates for bootstrap resampling.
    CI_method         - String. 'log', 'cbrt' or 'none' depending on method used
                        for evaluating confidence interval (bootstrapping and
                        jackknifing only).
    forward           - Boolean. True if CUtIe is run in the forward direction,
                        False if reverse.
    forward_stats     - List of strings. Contains list of statistics e.g. 'kpc'
                        'jkp' that pertain to forward (non-reverse) CUtIe
                        analysis.
    reverse_stats     - List of strings. Contains list of statistics e.g. 'rpc'
                        'rjkp' that pertain to reverse CUtIe analysis.
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    mine_str          - 2D array. Entry in i-th row, j-th column corresponds to
                        MIC strength between var i and var j.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.

    OUTPUTS
    new_rev_corr      - Indicator array of length n_samp indicating whether that
                        sample caused the correlation to reverse sign (1) or
                        not (0).
    new_truths        - Indicator array of length n_samp indicating whether that
                        sample caused the correlation to be classifed as a
                        CUtIe (1) or not (0).
    extrema_p         - Float. Lowest or highest p value observed thusfar for a
                        particular correlation, depending if reverse or forward
                        CUtIe was run, respectively.
    extrema_r         - Float. Highest or lowest R value observed thusfar for a
                        particular correlation, depending if reverse or forward
                        CUtIe was run, respectively.

    """
    # CUtIe
    if statistic in ['kpc', 'rpc', 'ksc', 'rsc', 'mine', 'rmine', 'kkc', 'rkc']:
        new_rev_corr, new_truths, extrema_p, extrema_r = resamplek_cutie(
            var1, var2, n_samp, samp_var1, samp_var2, pvalues, threshold,
            index + 1, sign, forward, statistic, fold, fold_value, pvalue_bins,
            mine_str, mine_bins)

    # jackknife
    elif statistic in ['jkp', 'rjkp', 'jks', 'rjks', 'jkm', 'rjkm', 'jkk', 'rjkk']:
        new_rev_corr, new_truths, extrema_p, extrema_r = jackknifek_cutie(
            var1, var2, n_samp, samp_var1, samp_var2, pvalues, threshold,
            index + 1, sign, forward, statistic, CI_method, pvalue_bins,
            mine_str, mine_bins)

    # bootstrap
    elif statistic in ['bsp', 'rbsp', 'bss', 'rbss', 'bsm', 'rbsm', 'bsk', 'rbsk']:
        new_rev_corr, new_truths, extrema_p, extrema_r = bootstrap_cutie(
            var1, var2, n_samp, samp_var1, samp_var2, pvalues, threshold, sign,
            forward, statistic, CI_method, n_replicates, pvalue_bins, mine_str,
            mine_bins)

    else:
        raise ValueError('Invalid statistic chosen %s' % statistic)

    # obtain most extreme p and R-sq values
    if forward:
        extrema_p = np.max(extrema_p)
        extrema_r = np.min(extrema_r)
    elif not forward:
        extrema_p = np.min(extrema_p)
        extrema_r = np.max(extrema_r)

    return new_rev_corr, new_truths, extrema_p, extrema_r

def compute_pc(new_var1, new_var2):
    """
    Compute Pearson correlation and return p and r values.
    INPUTS
    ----------------------------------------------------------------------------
    new_var1 - Array. Length sample size containing observations for given
               variable from file 1.
    new_var2 - Array. Same as new_var1 but for file 2.
    """
    # if resulting variables do not contain enough points
    if new_var1.size < 2 or new_var2.size < 2:
        p_value = 1
        r_value = 0

    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            new_var1, new_var2)

    # if p_value is nan
    if np.isnan(p_value):
        p_value = 1
        r_value = 0

    return p_value, r_value


def compute_sc(new_var1, new_var2):
    """
    Compute Spearman correlation and return p and r values.
    ----------------------------------------------------------------------------
    INPUTS
    new_var1 - Array. Length sample size containing observations for given
               variable from file 1.
    new_var2 - Array. Same as new_var1 but for file 2.
    """

    # if resulting variables do not contain enough points
    if new_var1.size < 2 or new_var2.size < 2:
        p_value = 1
        r_value = 0

    else:
        r_value, p_value = stats.spearmanr(new_var1, new_var2)

    # if p_value is nan
    if np.isnan(p_value):
        p_value = 1
        r_value = 0

    return p_value, r_value


def compute_kc(new_var1, new_var2):
    """
    Compute Kendall correlation and return p and r values.
    ----------------------------------------------------------------------------
    INPUTS
    new_var1 - Array. Length sample size containing observations for given
               variable from file 1.
    new_var2 - Array. Same as new_var1 but for file 2.
    """

    # if resulting variables do not contain enough points
    if new_var1.size < 2 or new_var2.size < 2:
        p_value = 1
        r_value = 0

    else:
        r_value, p_value = stats.kendalltau(new_var1, new_var2)

    # if p_value is nan
    if np.isnan(p_value):
        p_value = 1
        r_value = 0

    return p_value, r_value


###
# MINE specific handling
###

def compute_mine(new_var1, new_var2, pvalue_bins, mine_str, mine_bins):
    """
    Compute MINE correlation and return p and r values. Defaults are used from
    MINE API's page.
    https://minepy.readthedocs.io/en/latest/python.html
    ----------------------------------------------------------------------------
    INPUTS
    new_var1          - Array. Length sample size containing observations for
                        given variable from file 1.
    new_var2          - Array. Same as new_var1 but for file 2.
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    mine_str          - 2D array. Entry in i-th row, j-th column corresponds to
                        MIC strength between var i and var j.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.
    """

    # if resulting variables do not contain enough points
    if new_var1.size < 2 or new_var2.size < 2:
        p_value = 1
        r_value = 0
    else:
        data = np.stack([new_var1, new_var2], 0)
        r_value = minepy.pstats(data, alpha=0.6, c=10, est="mic_approx")[0]
        p_value = str_to_pvalues(pvalue_bins, r_value, mine_bins)

    return p_value, r_value

def str_to_pvalues(pvalue_bins, mine_str, mine_bins):
    """
    Convert MIC_str to pvalue using parsed table from MINE.
    ----------------------------------------------------------------------------
    INPUTS
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    mine_str          - 2D array. Entry in i-th row, j-th column corresponds to
                        MIC strength between var i and var j.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.
    mine_p            - 2D np array where each entry corresponds to MINE pvalue
                        of the correlation between var i and var j.
    """
    found, midpoint = binarySearchBins(pvalue_bins, mine_str)
    if found:
        mine_p = mine_bins[midpoint][1]
    else:
        mine_p = 1

    return mine_p

def binarySearchBins(alist, item):
    """
    FUNCTION
    Takes an item (float) and returns the index (midpoint) and boolean (found)
    of the bin (alist) in which item belongs
    """
    first = 0
    last = len(alist)-1
    found = False
    copy = [1]
    copy.extend(alist[:])
    copy.append(0)
    midpoint = 0
    while first <= last and not found:
        midpoint = (first + last)//2
        if copy[midpoint] >= item and copy[midpoint + 1] < item:
            found = True
        else:
            if item > copy[midpoint] and item > copy[midpoint + 1]:
                last = midpoint-1
            else:
                first = midpoint+1

    return found, midpoint


def update_rev_extrema_rp(sign, r_value, p_value, indices, reverse, extrema_p,
                          extrema_r, forward=True):
    """
    Check sign, r and p value and update reverse, maxp, and minr
    ----------------------------------------------------------------------------
    INPUTS
    sign      - Integer. -1 or 1, depending on original sign of correlation to
                check against following re-evaluation.
    r_value   - Same as p_value but for R / correlation strength.
    p_value   - Float. P-value computed for a given correlation on a specific
                iteration of CUtIe upon removing a subset of points.
    indices   - Set of integers. Subset of samples (size k, usually k = 1 for
                one point CUtIe) being removed.
    extrema_p - 1D array. Length n_samp, contains lowest or highest p value
                observed thusfar for a particular sample, depending if reverse
                or forward CUtIe was run, respectively across all i in {1,...,k}
                iterations of CUtIe_k.
    extrema_r - 1D array. Same as extrema_p but for R / correlation strength
                values.
    forward   - Boolean. True if CUtIe is run in the forward direction, False if
                reverse.

    OUTPUT (in addition to above)
    reverse   - 1D array. Indicator array where entry is 1 if that i-th sample
                caused the correlation to change sign.
    """
    # if sign has reversed
    if np.sign(r_value) != sign and np.sign(r_value) != np.nan:
        for i in indices:
            reverse[i] += 1

    # update most extreme p and r values
    if forward is True:
        for i in indices:
            if p_value > extrema_p[i]:
                extrema_p[i] = p_value
            if np.absolute(r_value) < np.absolute(extrema_r[i]):
                extrema_r[i] = r_value
    elif forward is False:
        for i in indices:
            if p_value < extrema_p[i]:
                extrema_p[i] = p_value
            if np.absolute(r_value) > np.absolute(extrema_r[i]):
                extrema_r[i] = r_value

    return reverse, extrema_p, extrema_r


def jackknifek_cutie(var1_index, var2_index, n_samp, samp_var1, samp_var2,
                     pvalues, threshold, resample_k, sign, forward, statistic,
                     CI_method, pvalue_bins, mine_str, mine_bins):
    """
    Perform jackknife resampling on a given pair of variables and test CUtIe
    status via confidence based interval methods.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index        - Integer. Index of variable in file 1.
    var2_index        - Integer. Index of variable in file 2.
    n_samp            - Integer. Number of samples.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids when parsed.
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    pvalues           - 2D array. Entry row i, col j represents p value of
                        correlation between i-th var1 and j-th var2.
    threshold         - Float. Level of significance testing (after adjusting
                        for multiple comparisons)
    resample_k        - Integer. Number of points being resampled by CUtIe.
    sign              - Integer. -1 or 1, depending on original sign of
                        correlation to check against following re-evaluation.
    forward           - Boolean. True if CUtIe is run in the forward direction,
                        False if reverse.
    statistic         - String. Describes analysis being performed.
    CI_method         - String. 'log', 'cbrt' or 'none' depending on method used
                        for evaluating confidence interval (bootstrapping and
                        jackknifing only).
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    mine_str          - 2D array. Entry in i-th row, j-th column corresponds to
                        MIC strength between var i and var j.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.

    OUTPUTS
    reverse           - 1D array. Index i is 1 if the correlation changes sign
                        upon removing sample i.
    exceeds           - 1D array. Index i is 1 if removing that sample causes
                        the correlation to become insignificant in at least 1
                        different pairwise correlations
    extrema_p         - 1D array. Length n_samp, contains lowest or highest p
                        value observed thusfar for a particular sample,
                        depending if reverse or forward CUtIe was run
                        respectively across all i in {1,...,k} iterations of
                        CUtIe_k.
    extrema_r         - 1D array. Same as extrema_p but for R / correlation
                        strength values.
    """

    # initialize indicators and variables
    exceeds, reverse, extrema_p, extrema_r, var1, var2 = utils.init_var_indicators(
        var1_index, var2_index, samp_var1, samp_var2, forward)

    p_values = []
    # iteratively delete k samples and recompute statistics
    combs = [list(x) for x in itertools.combinations(range(n_samp), resample_k)]
    for indices in combs:
        new_var1 = var1[~np.in1d(range(len(var1)), indices)]
        new_var2 = var2[~np.in1d(range(len(var2)), indices)]

        # remove NaNs
        new_var1, new_var2 = utils.remove_nans(new_var1, new_var2)

        # compute new p_value and r_value depending on statistic
        if statistic == 'jkp' or statistic == 'rjkp':
            p_value, r_value = compute_pc(new_var1, new_var2)
        elif statistic == 'jks' or statistic == 'rjks':
            p_value, r_value = compute_sc(new_var1, new_var2)
        elif statistic == 'jkk' or statistic == 'rjkk':
            p_value, r_value = compute_kc(new_var1, new_var2)
        elif statistic == 'jkm' or statistic == 'rjkm':
            p_value, r_value = compute_mine(new_var1, new_var2, pvalue_bins,
                                            mine_str, mine_bins)

        # update reverse, maxp, and minr
        reverse, extrema_p, extrema_r = update_rev_extrema_rp(
            sign, r_value, p_value, indices, reverse, extrema_r, extrema_p,
            forward)
        p_values.append(p_value)


    # generate log confidence interval on p-value
    CI, p_mu, p_sigma = get_pCI(np.array(p_values), n_samp, CI_method)

    # test confidence interval
    if forward is True:
        exceeds = test_CI(
            CI, threshold, exceeds,
            [item for sublist in combs for item in sublist], True, CI_method)
    elif forward is False:
        exceeds = test_CI(
            CI, threshold, exceeds,
            [item for sublist in combs for item in sublist], False, CI_method)

    return reverse, exceeds, extrema_p, extrema_r


def bootstrap_cutie(var1_index, var2_index, n_samp, samp_var1, samp_var2,
                    pvalues, threshold, sign, forward, statistic, CI_method,
                    n_replicates, pvalue_bins, mine_str, mine_bins):

    """
    Perform bootstrap resampling on a given pair of variables and test CUtIe
    status via confidence based interval methods.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index        - Integer. Index of variable in file 1.
    var2_index        - Integer. Index of variable in file 2.
    n_samp            - Integer. Number of samples.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids when parsed.
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    pvalues           - 2D array. Entry row i, col j represents p value of
                        correlation between i-th var1 and j-th var2.
    threshold         - Float. Level of significance testing (after adjusting
                        for multiple comparisons)
    sign              - Integer. -1 or 1, depending on original sign of
                        correlation to check against following re-evaluation.
    forward           - Boolean. True if CUtIe is run in the forward direction,
                        False if reverse.
    statistic         - String. Describes analysis being performed.
    CI_method         - String. 'log', 'cbrt' or 'none' depending on method used
                        for evaluating confidence interval (bootstrapping and
                        jackknifing only).
    n_replicates      - Integer. Number of bootstrap samples.
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    mine_str          - 2D array. Entry in i-th row, j-th column corresponds to
                        MIC strength between var i and var j.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.

    OUTPUTS
    reverse           - 1D array. Index i is 1 if the correlation changes sign
                        upon removing sample i.
    exceeds           - 1D array. Index i is 1 if removing that sample causes
                        the correlation to become insignificant in at least 1
                        different pairwise correlations
    extrema_p         - 1D array. Length n_samp, contains lowest or highest p
                        value observed thusfar for a particular sample,
                        depending if reverse or forward CUtIe was run
                        respectively across all i in {1,...,k} iterations of
                        CUtIe_k.
    extrema_r         - 1D array. Same as extrema_p but for R / correlation
                        strength values.
    """
    # initialize indicators and variables
    exceeds, reverse, extrema_p, extrema_r, var1, var2 = utils.init_var_indicators(
        var1_index, var2_index, samp_var1, samp_var2, forward)

    p_values = []

    for k in range(n_replicates):
        new_samp = np.random.choice(range(n_samp), size=n_samp, replace=True)
        new_var1 = []
        new_var2 = []
        for j in range(n_samp):
            new_var1.append(var1[new_samp[j]])
            new_var2.append(var2[new_samp[j]])


        # remove NaNs
        new_var1, new_var2 = utils.remove_nans(new_var1, new_var2)

        # compute new p_value and r_value depending on statistic
        if statistic == 'bsp' or statistic == 'rbsp':
            p_value, r_value = compute_pc(new_var1, new_var2)
        elif statistic == 'bss' or statistic == 'rbss':
            p_value, r_value = compute_sc(new_var1, new_var2)
        elif statistic == 'bsk' or statistic == 'rbsk':
            p_value, r_value = compute_kc(new_var1, new_var2)
        elif statistic == 'bsm' or statistic == 'rbsm':
            p_value, r_value = compute_mine(new_var1, new_var2, pvalue_bins,
                                            mine_str, mine_bins)

        # update reverse, maxp, and minr
        reverse, extrema_p, extrema_r = update_rev_extrema_rp(
            sign, r_value, p_value, range(n_samp), reverse, extrema_p,
            extrema_r)

        p_values.append(p_value)

    CI, p_mu, p_sigma = get_pCI(np.array(p_values), n_samp, CI_method)

    if forward is True:
        exceeds = test_CI(
            CI, threshold, exceeds, range(n_samp), True, CI_method)
    elif forward is False:
        exceeds = test_CI(
            CI, threshold, exceeds, range(n_samp), False, CI_method)

    return reverse, exceeds, extrema_p, extrema_r

def resamplek_cutie(var1_index, var2_index, n_samp, samp_var1, samp_var2,
                    pvalues, threshold, resample_k, sign, forward, statistic,
                    fold, fold_value, pvalue_bins, mine_str, mine_bins):
    """
    Perform CUtIe resampling on a given pair of variables and test CUtIe
    status via confidence based interval methods.
    ----------------------------------------------------------------------------
    INPUTS
    var1_index        - Integer. Index of variable in file 1.
    var2_index        - Integer. Index of variable in file 2.
    n_samp            - Integer. Number of samples.
    samp_var1         - 2D array. Each value in row i col j is the level of
                        variable j corresponding to sample i in the order that
                        the samples are presented in samp_ids when parsed.
    samp_var2         - 2D array. Same as samp_var1 but for file 2.
    pvalues           - 2D array. Entry row i, col j represents p value of
                        correlation between i-th var1 and j-th var2.
    threshold         - Float. Level of significance testing (after adjusting
                        for multiple comparisons)
    sign              - Integer. -1 or 1, depending on original sign of
                        correlation to check against following re-evaluation.
    forward           - Boolean. True if CUtIe is run in the forward direction,
                        False if reverse.
    statistic         - String. Describes analysis being performed.
    fold              - Boolean. Determines whether you require the new P value
                        to be a certain fold greater to be classified as a CUtIe.
    fold_value        - Float. Determines fold difference constraint imposed on
                        the resampled p-value needed for a correlation to be
                        classified as a CUtIe.
    pvalue_bins       - List. Sorted list of pvalues from greatest to least used
                        by MINE to bin the MIC_str.
    mine_str          - 2D array. Entry in i-th row, j-th column corresponds to
                        MIC strength between var i and var j.
    mine_bins         - 2D Array. Obtained from parse_minep. Each row is in
                        format [MIC_str, pvalue, stderr of pvalue]. Pvalue
                        corresponds to probability of observing MIC_str as or
                        more extreme as observed MIC_str.

    OUTPUTS
    reverse           - 1D array. Index i is 1 if the correlation changes sign
                        upon removing sample i.
    exceeds           - 1D array. Index i is 1 if removing that sample causes
                        the correlation to become insignificant in at least 1
                        different pairwise correlations
    extrema_p         - 1D array. Length n_samp, contains lowest or highest p
                        value observed thusfar for a particular sample,
                        depending if reverse or forward CUtIe was run
                        respectively across all i in {1,...,k} iterations of
                        CUtIe_k.
    extrema_r         - 1D array. Same as extrema_p but for R / correlation
                        strength values.
    """
    # initialize indicators and variables
    exceeds, reverse, extrema_p, extrema_r, var1, var2 = utils.init_var_indicators(
        var1_index, var2_index, samp_var1, samp_var2, forward)

    # iteratively delete k samples and recompute statistics
    combs = [list(x) for x in itertools.combinations(range(n_samp), resample_k)]
    for indices in combs:
        new_var1 = var1[~np.in1d(range(len(var1)), indices)]
        new_var2 = var2[~np.in1d(range(len(var2)), indices)]

        # remove NaNs
        new_var1, new_var2 = utils.remove_nans(new_var1, new_var2)

        # compute new p_value and r_value depending on statistic
        if statistic == 'kpc' or statistic == 'rpc':
            p_value, r_value = compute_pc(new_var1, new_var2)
        elif statistic == 'ksc' or statistic == 'rsc':
            p_value, r_value = compute_sc(new_var1, new_var2)
        elif statistic == 'kkc' or statistic == 'rkc':
            p_value, r_value = compute_kc(new_var1, new_var2)
        elif statistic == 'mine' or statistic == 'rmine':
            p_value, r_value = compute_mine(new_var1, new_var2, pvalue_bins,
                                            mine_str, mine_bins)

        # update reverse, maxp, and minr
        reverse, extrema_p, extrema_r = update_rev_extrema_rp(
            sign, r_value, p_value, indices, reverse, extrema_p, extrema_r,
            forward)

        # check sign reversal
        if np.sign(r_value) != sign:
            for i in indices:
                reverse[i] += 1

        if forward is True:
            # fold change p-value restraint
            if fold:
                if (p_value > threshold and \
                    p_value > pvalues[var1_index][var2_index] * fold_value) or \
                    np.isnan(p_value):
                    for i in indices:
                        exceeds[i] += 1
            elif p_value > threshold or np.isnan(p_value):
                for i in indices:
                    exceeds[i] += 1

        elif forward is False:
            # fold change p-value restraint
            if fold:
                if (p_value < threshold and \
                    p_value < pvalues[var1_index][var2_index] * fold_value) or \
                    np.isnan(p_value):
                    for i in indices:
                        exceeds[i] += 1
            elif p_value < threshold or np.isnan(p_value):
                for i in indices:
                    exceeds[i] += 1

    return reverse, exceeds, extrema_p, extrema_r

###
# Confidence interval handling
###

def get_pCI(p_values, n_samp, CI_method='log', zero_replace=10e-100):
    """
    Compute logp confidence interval.
    ----------------------------------------------------------------------------
    INPUTS
    pvalues      - 2D array. Entry row i, col j represents p value of correlation
                   between i-th var1 and j-th var2.
    n_samp       - Integer. Number of samples.
    CI_method    - String. 'log', 'cbrt' or 'none' depending on method used for
                   evaluating confidence interval (bootstrapping and jackknifing
                   only).
    zero_replace - Float. Value to replace 0's with in p-value array.

    OUTPUTS
    pCI          - Tuple. Lower and upper bounds of 95 percent CI.
    p_mu         - Float. Mean of CI.
    p_sigma      - Float. SDev of CI.
    """

    CI_method = str(CI_method)
    if CI_method == 'log':
        p_values[p_values == 0] = zero_replace
        logp_values = np.log(p_values)
        p_mu = np.mean(logp_values)
        p_sigma = np.std(logp_values)
    elif CI_method == 'cbrt':
        cbrtp_values = np.cbrt(p_values)
        p_mu = np.mean(cbrtp_values)
        p_sigma = np.std(cbrtp_values)
    elif CI_method == 'none':
        p_mu = np.mean(p_values)
        p_sigma = np.std(p_values)

    pCI = (p_mu - 1.96 * p_sigma / np.sqrt(n_samp),
           p_mu + 1.96 * p_sigma / np.sqrt(n_samp))

    return pCI, p_mu, p_sigma

def test_CI(CI, threshold, exceeds, indices, upper=True, CI_method='log'):
    """
    Test if lower bound of CI is below a threshold and update exceeds indicator
    matrix.
    ----------------------------------------------------------------------------
    INPUTS
    CI          - Tuple. Lower and upper bounds of 95 percent CI.
    threshold   - Float. Level of significance testing (after adjusting for
                  multiple comparisons)
    exceeds     - 1D array. Index i is 1 if removing that sample causes the
                  correlation to become insignificant in at least 1 different
                  pairwise correlations
    indices     - Set of integers. Subset of samples (size k, usually k = 1 for
                  one point CUtIe) being removed.
    upper       - Boolean. True if testing if the upper bound of the CI exceeds
                  a value, false if testing if the lower bound of the CI falls
                  below a value.
    CI_method   - String. 'log', 'cbrt' or 'none' depending on method used for
                  evaluating confidence interval (bootstrapping and jackknifing
                  only).
    """
    CI_method = str(CI_method)
    if CI_method == 'log':
        if upper and CI[1] > np.log(threshold):
            for i in indices:
                exceeds[i] += 1
        elif not upper and CI[0] < np.log(threshold):
            for i in indices:
                exceeds[i] += 1
    elif CI_method == 'cbrt':
        if upper and CI[1] > np.cbrt(threshold):
            for i in indices:
                exceeds[i] += 1
        elif not upper and CI[0] < np.cbrt(threshold):
            for i in indices:
                exceeds[i] += 1
    elif CI_method == 'none':
        if upper and CI[1] > threshold:
            for i in indices:
                exceeds[i] += 1
        elif not upper and CI[0] < threshold:
            for i in indices:
                exceeds[i] += 1
    return exceeds
