#!/usr/bin/env python
from __future__ import division

import numpy as np
from collections import defaultdict
import itertools

from cutie import parse
from cutie import output



def indicator(n_var1, n_var2, initial_corr, true_corr):
    """
    Takes in lists of initial pairs of variables and true correlations and
    returns a matrix indicating TN, FP and TP/FN (indicator matrix). Entry i,j
    is:
        0  if var1 i, var2 j were never significantly correlated (TN)
        -1 if var1 i, var2 j were falsely correlated (FP/TN)
        1  if var1 i, var2 j is correlated following resampling (TP/FN)
    ----------------------------------------------------------------------------
    INPUTS
    n_var1       - Integer. Number of variables in file 1.
    n_var2       - Integer. Number of variables in file 2.
    initial_corr - Set of tuples. Each tuple is a variable pair that is in the
                   set of correlations deemed initially significant or
                   insignificant (in CUtIe or reverse CUtIe, respectively)
                   prior to resampling
    true_corr     - Set of tuples for a given k referring to variable pairs
                   deemed true correlations following resampling of k points
                   according to CUtie.

    OUTPUTS
    indicators    - 2D array. Size (n_var1 x n_var2) where each i,j entry is
                    described as above.
    """
    indicators = np.zeros((n_var1, n_var2))
    for point in initial_corr:
        i, j = point
        indicators[i][j] = -1
    for point in true_corr:
        i, j = point
        indicators[i][j] = 1
    return indicators


def init_var_indicators(var1_index, var2_index, samp_var1, samp_var2, forward):
    """
    Initialize indicator matrices and variable matrices
    ----------------------------------------------------------------------------
    INPUTS
    var1_index - Integer. Index of variable from file 1 for pairwise correlation.
    var2_index - Integer. Index of variable from file 1 for pairwise correlation.
    samp_var1  - 2D array. Each value in row i col j is the level of variable j
                 corresponding to sample i in the order that the samples are
                 presented in samp_ids.
    samp_var2  - 2D array. Same as samp_var1 but for file 2.
    forward    - Boolean. True if CUtIe is run in the forward direction, False if
                 reverse.

    OUTPUT (in addition to above)
    reverse    - 1D array. Index i is 1 if the correlation changes sign upon
                 removing sample i.
    exceeds    - 1D array. Index i is 1 if removing that sample causes the
                 correlation to become insignificant in at least 1 different
                 pairwise correlations
    extrema_p - 1D array. Length n_samp, contains lowest or highest p value
                observed thusfar for a particular sample, depending if reverse
                or forward CUtIe was run, respectively across all i in {1,...,k}
                iterations of CUtIe_k.
    extrema_r - 1D array. Same as extrema_p but for R / correlation strength
                values.
    var1      - 1D array. Values for specified variable (from var_index1) from
                file 1.
    var2      - 1D array. Values for specified variable (from var_index2) from
                file 2.
    """
    n_var1, n_var2, n_samp = get_param(samp_var1, samp_var2)

    exceeds = np.zeros(n_samp)
    reverse = np.zeros(n_samp)
    if forward is True:
        extrema_p = np.zeros(n_samp)
        extrema_r = np.ones(n_samp)
    elif forward is False:
        extrema_p = np.ones(n_samp)
        extrema_r = np.zeros(n_samp)

    # slice relevant variables
    var1 = samp_var1[:, var1_index]
    var2 = samp_var2[:, var2_index]
    return exceeds, reverse, extrema_p, extrema_r, var1, var2

def return_indicators(n_var1, n_var2, initial_corr, true_corr, resample_k):
    """
    Construct indicator matrices for keeping track of false_corrs.
    ----------------------------------------------------------------------------
    INPUTS
    n_var1       - Integer. Number of variables in file 1.
    n_var2       - Integer. Number of variables in file 2.
    initial_corr - Set of tuples. Each tuple is a variable pair that is in the
                   set of correlations deemed initially significant or
                   insignificant (in CUtIe or reverse CUtIe, respectively)
                   prior to resampling
    true_corr    - Set of tuples for a given k referring to variable pairs
                   deemed true correlations following resampling of k points
                   according to CUtie.
    resample_k   - Integer. Number of points being resampled by CUtIe.

    OUTPUTS
    indicators   - Dictionary. Key is the number of points removed and entry i j
                   corresponds to indicator value for correlation between var1 i
                   and var2 j.
    """
    indicators = {}
    for i in range(resample_k):
        indicators[str(i+1)] = indicator(n_var1, n_var2, initial_corr,
                                         true_corr[str(i+1)])

    return indicators

def remove_nans(var1, var2):
    """
    Remove Nan Points (at least one Nan in an observation).
    ----------------------------------------------------------------------------
    INPUTS
    var1      - 1D array. Values for specified variable (from var_index1) from
                file 1.
    var2      - 1D array. Values for specified variable (from var_index2) from
                file 2.

    Example:
    # remove all points where one or both values are NAN
    # new_var1 = np.array([1,2,np.nan])
    # new_var2 = np.array([1,np.nan,3])
    # stacked = array([[  1.,   2.,  nan],
    #                  [  1.,  nan,   3.]])
    # np.isnan(stacked) = array([[False, False,  True],
    #                             [False,  True, False]], dtype=bool)
    # np.all(~np.isnan(stacked), axis = 0) = array([ True, False, False])
    # stacked[:,np.all(~np.isnan(stacked), axis = 0)] =  array([[ 1.],
    #                                                           [ 1.]])
    """
    stacked = np.stack([var1, var2], 0)
    stacked = stacked[:, np.all(~np.isnan(stacked), axis=0)]
    new_var1 = stacked[0]
    new_var2 = stacked[1]
    return new_var1, new_var2


def get_param(samp_var1, samp_var2):
    """
    Commonly used helper function. Extracts, n_var1, n_var2 and n_samp from
    samp_var1 and samp_var2 matrices.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var1 - 2D array. Each value in row i col j is the level of variable j
                corresponding to sample i in the order that the samples are
                presented in samp_ids.
    samp_var2 - 2D array. Same as samp_var1 but for file 2.

    OUTPUTS
    n_var1    - Integer. Number of variables in file 1.
    n_var2    - Integer. Number of variables in file 2.
    n_samp    - Integer. Number of samples (should be same between file 1 and 2)
    """
    n_var1 = np.size(samp_var1, 1)
    n_var2 = np.size(samp_var2, 1)
    n_samp = np.size(samp_var1, 0)

    return n_var1, n_var2, n_samp

def calculate_intersection(names, sets, log_fp):
    """
    Calculates all possible intersection (i.e. Venn Diagram) of sets.
    ----------------------------------------------------------------------------
    INPUTS
    names        - List. Strings referring to names of sets.
    sets         - List of sets. Order of sets must match order of sets in names.
    log_fp       - String. File path of log file.

    e.g.
    names = ['a','b','c']
    sets = [set_a, set_b, set_c] where set_a = set([(1,2),(1,3)]),
        set_b = set([(1,2),(1,4)]) etc.

    OUTPUTS
    region_combs - List of strings. Each string describes a region (union) in the
                   Venn Diagram.
    region_sets  - Dictionary. Maps key (region on Venn Diagram) to elements in
                   that set (e.g. variable pairs)

    """
    # temporary mapping of name  to set
    name_to_set = {}
    for i in range(len(names)):
        name_to_set[names[i]] = sets[i]

    # get regions and initialize default dict of list
    region_combs = []
    for i in range(1, len(names)+1):
        els = [list(x) for x in itertools.combinations(names, i)]
        region_combs.extend(els)
    region_sets = defaultdict(list)

    # create union of sets
    union_set = set()
    for indiv_set in sets:
        union_set = union_set.union(indiv_set)

    # for each region, determine in_set and out_set
    for region in region_combs:
        in_set = set(region)
        out_set = set(names).difference(in_set)

        # for each in_set,
        final_set = union_set
        for in_s in in_set:
            final_set = final_set.intersection(name_to_set[in_s])

        for out_s in out_set:
            final_set = final_set.difference(name_to_set[out_s])

        region_sets[str(region)] = final_set

        output.write_log('The amount of unique elements in set ' + \
            str(region) + ' is ' + str(len(final_set)), log_fp)

    return region_combs, region_sets


def initialize_stat_dicts(resample_k, n_var1, n_var2, statistic, forward_stats,
                          reverse_stats):
    """
    Create empty dicts for keeping track of CUtIe analysis.
    ----------------------------------------------------------------------------
    INPUTS
    resample_k        - Integer. Number of points being resampled by CUtIe.
    n_var1            - Integer. Number of variables in file 1.
    n_var2            - Integer. Number of variables in file 2.
    statistic         - String. Describes analysis being performed.

    forward_stats     - List of strings. Contains list of statistics e.g. 'kpc'
                        'jkp' that pertain to forward (non-reverse) CUtIe
                        analysis.
    reverse_stats     - List of strings. Contains list of statistics e.g. 'rpc'
                        'rjkp' that pertain to reverse CUtIe analysis.

    OUTPUTS
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
    """
    # create dicts of points to track true_sig and reversed-sign correlations
    true_corr = {}
    true_comb_to_rev = {}
    false_comb_to_rev = {}

    # create matrices dict to hold the most extreme values of p and r (for R-sq)
    extrema_p = {}
    extrema_r = {}

    # initialize dictionary entries as empty lists
    for i in range(resample_k):
        true_corr[str(i+1)] = []
        true_comb_to_rev[str(i+1)] = []
        false_comb_to_rev[str(i+1)] = []
        if statistic in forward_stats:
            extrema_p[str(i+1)] = np.ones([n_var1, n_var2])
            extrema_r[str(i+1)] = np.zeros([n_var1, n_var2])
        elif statistic in reverse_stats:
            extrema_p[str(i+1)] = np.zeros([n_var1, n_var2])
            extrema_r[str(i+1)] = np.ones([n_var1, n_var2])

    return true_corr, true_comb_to_rev, false_comb_to_rev, extrema_p, extrema_r

