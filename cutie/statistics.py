#!/usr/bin/env python
from __future__ import division
    
import os
import math
import itertools    
import numpy as np
from scipy import stats
import statsmodels.api as sm
from collections import defaultdict

from cutie import parse
from cutie import output

def initial_stats_SLR(n_var1, n_var2, samp_var1, samp_var2, functions, mapf, 
                        f_stats, rm_zero = False):
    """ 
    INPUTS
    n_var1:    number of var1s
    n_var2:    number of var2s
    samp_var1: np array where row i col j corresponds to level of var1 j in 
               sample i
    samp_var2: np array where row i col j corresponds to level of var2 j in 
               sample i                  
    functions: list of strings of function names 
    mapf:      dict that maps function name to function object
    f_stats:   dict that maps function name to list of output strings
    rm_zero:   remove values that are 0 in both x and y

    OUTPUTS
    statistics: list of dict where each dict corresponds to each element 
                in function in the order they are passed and where each 
                key corresponds to a particular statistic calculated by 
                each function in functions and the element is an array 
                where row i col j corresponds to the value of a given 
                statistic to the values for var1 row i and var2 col j.
     
    FUNCTION
    Function that computes an initial set of statistics per the specified 
    functions. Returns a dict where the key is a statistical function and  
    the element is an initial matrix with dimensions n_rel_stats x n_var1 x 
    n_var2, corresponding to the relevant statistics from simple linear 
    regression (SLR) between each var1 and var2. 
    """
    stat_dict = {}
    
    # retrieve relevant stats and create dictionary entry, 3D array
    for f in functions:
        rel_stats = f_stats[f]
        stat_dict[f] = np.zeros((len(rel_stats), 
                                 n_var1, 
                                 n_var2))

    # subset the data matrices into the cols needed
    for var1 in xrange(n_var1):
        for var2 in xrange(n_var2):
            var1_values = samp_var1[:,var1]
            var2_values = samp_var2[:,var2] 
            # remove zero values
            stacked = np.stack([var1_values,var2_values],0)
            if rm_zero is True:
                stacked = stacked[:,~np.all(stacked == 0.0, axis = 0)]
            # remove NANs
            stacked = stacked[:,np.all(~np.isnan(stacked), axis = 0)]
            var1_values = stacked[0]
            var2_values = stacked[1]
            for f in functions:
                # values is a list of the relevant_stats in order
                if len(var1_values) == 0 or len(var2_values) == 0: 
                    values = np.zeros([len(f_stats)])
                    values[:] = np.nan
                else:
                    values = mapf[f](var1_values, var2_values)
                for s in xrange(len(values)):
                    stat_dict[f][s][var1][var2] = values[s] 
    
    return stat_dict 

def assign_statistics(n_var1, n_var2, samp_var1, samp_var2, statistic, 
                        rm_zero = False):
    """
    n_var1:    number of var1s
    n_var2:    number of var2s
    samp_var1: np array where row i col j corresponds to level of var1 j in 
               sample i
    samp_var2: np array where row i col j corresponds to level of var2 j in 
               sample i        
    statistic: statistic of choice (e.g. kpc)
    rm_zero:   boolean whether zeros should be removed
    """
    if statistic == 'kpc':
        functions = ['stats.linregress']
        mapf = {'stats.linregress': stats.linregress}
        f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','ppvalue','stderr']}
        stat_dict = initial_stats_SLR(n_var1, n_var2, samp_var1,samp_var2, 
                                     functions,
                                     mapf,
                                     f_stats)
        
        pvalues = stat_dict['stats.linregress'][3]
        correlations = stat_dict['stats.linregress'][2]
        logpvals = np.log(stat_dict['stats.linregress'][3])
        r2vals = np.square(stat_dict['stats.linregress'][2])

        return pvalues, correlations, logpvals, r2vals

    elif statistic == 'ksc':
        functions = ['stats.spearmanr']
        mapf = {'stats.spearmanr': stats.spearmanr}
        f_stats = {'stats.linregress': 
               ['b1', 'b0', 'pcorr','ppvalue','stderr'],
           'stats.spearmanr':
               ['scorr','spvalue']}

        stat_dict = initial_stats_SLR(n_var1, n_var2, samp_var1, samp_var2, 
                                     functions,
                                     mapf,
                                     f_stats)
        
        pvalues = stat_dict['stats.spearmanr'][1]
        logpvals = np.log(pvalues)
        correlations = stat_dict['stats.spearmanr'][0]

        return pvalues, logpvals, correlations

    else:
        print 'Invalid statistic chosen'

    return

def set_threshold(pvalues, alpha, mc, paired = False):
    """
    INPUTS
    pvalues: 2D np array of pvalues
    alpha:   float of original cutoff
    mc:      form of multiple corrections to use
             nomc: none
             bc: bonferroni
             fwer: family-wise error rate 
             fdr: false discovery rate
    paired:  boolean, true if correlations are between a single matrix 

    OUTPUTS
    threshold: float cutoff of pvalues

    FUNCTION
    Performs a multiple comparisons correction on the alpha value
    """
    print 'The type of mc correction used was ' + mc
    pvalues_copy = np.copy(pvalues)
    if paired == True:
        # fill the upper diagonal with nan as to not double count pvalues in FDR
        pvalues_copy[np.triu_indices(pvalues_copy.shape[1],0)] = np.nan
        # currently computing all pairs double counting
        n_corr = int(np.size(pvalues_copy,1) * (np.size(pvalues_copy,1) - 1))#/2)
    else:
        n_corr = np.size(pvalues_copy,0) * np.size(pvalues_copy,1)

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
        cn = 1.0
        thresholds = np.array([(float(k+1))/(len(pvalues_copy)) \
            * alpha/cn for k in xrange(len(pvalues_copy))])
        compare = np.where(pvalues_copy <= thresholds)[0]
        if len(compare) is 0:
            threshold = alpha
            print 'Warning: no p-values below threshold, defaulted with min(p) = ' \
                + str(min(pvalues_copy))
        else:
            threshold = thresholds[max(compare)]

    print 'The threshold value was ' + str(threshold)
    return threshold, n_corr

def indicator(n_var1, n_var2, initial_sig, true_sig):
    """
    INPUTS
    n_var1:      number of var1
    n_var2:      number of var2
    initial_sig: list of var1,var2 points that were significant correlations
                 prior to resampling
    true_sig:    list of var1,var2 points that remain significant following
                 resampling

    OUTPUTS
    indicators: array of size (n_var1 x n_var2) where each i,j entry is
                0  if var1 i, var2 j were never significantly correlated (TN)
                -1 if var1 i, var2 j were falsely correlated (FP)
                1  if var1 i, var2 j remain correlated following resampling (TP)

    FUNCTION
    Takes in lists of initially significant points and truly significant points
    and returns a matrix indicating TN, FP and TP
    """
    indicators = np.zeros((n_var1,n_var2))
    for point in initial_sig:
        i,j = point
        indicators[i][j] = -1
    for point in true_sig:
        i,j = point
        indicators[i][j] = 1
    return indicators

def resample1_cutie_pc(var1_index, var2_index, n_samp, n_var1, n_var2, 
                        samp_var1, samp_var2, influence1, influence2, threshold, 
                        sign, fold):
    """     
    INPUTS
    var1_index: integer of bacteria (in bact_names) to be evaluated
    var2_index: integer of metabolite (in meta_names) to be evaluated
    n_samp:     sample size
    n_var1:     number of var1
    n_var2:     number of var2
    samp_var1:  np array where row i col j indicates level of bact j 
                for sample i
    samp_var2:  np array where row i col j indicates level of meta j 
                for sample i
    influence1: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    influence2: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    threshold:  float of level of significance testing (after MC)
    sign:       original sign of correlation to check against following 
                re-evaluation
    fold:       boolean that determines whether you require the new P value to 
                be 100x greater to be labeled cutie

    OUTPUTS
    reverse:  array where index i is 1 if the correlation changes sign upon 
              removing sample i 
    exceeds:  array where index i is 1 if removing that sample causes the 
              correlation to become insignificant in at least 1 different 
              pairwise correlations
    corrs:    1D array of value of correlation strength with index i removed
    p_values: 1D array of value of pvalues with index i removed
    
    FUNCTION
    Takes a given var1 and var2 by indices and recomputes pearson correlation 
    by removing 1 out of n (sample_size) points from samp_ids. 
    Returns an indicator array where exceeds[i] is 1 if removing that sample causes
    the correlation to become insignificant in at least 1 different pairwise correlations
    """
    exceeds = np.zeros(n_samp)
    reverse = np.zeros(n_samp)
    corrs = np.zeros(n_samp)
    p_values = np.zeros(n_samp)
    var1_values = samp_var1[:,var1_index]
    var2_values = samp_var2[:,var2_index]
    
    # iteratively delete one sample and recompute statistics
    for sample_index in xrange(n_samp):
        new_var1_values = var1_values[~np.in1d(range(n_var1),sample_index)]
        new_var2_values = var2_values[~np.in1d(range(n_var2),sample_index)]

        if new_var1_values.size <= 3 or new_var2_values.size <= 3:
            p_value = 1
            r_value = 0
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                            new_var1_values, new_var2_values)
        
        # temporary boolean option to require a certain fold change in p value
        if fold:
            if (p_value > threshold and p_value > pvalue_matrix[var1_index][var2_index] \
                * 100.0) or np.isnan(p_value): 
                exceeds[sample_index] = 1
        elif p_value > threshold or np.isnan(p_value): 
            exceeds[sample_index] = 1
        if np.sign(r_value) != sign:
            reverse[sample_index] = 1

        corrs[sample_index] = r_value
        p_values[sample_index] = p_value

    return reverse, exceeds, corrs, p_values

def resample1_cutie_sc(var1_index, var2_index, n_samp, n_var1, n_var2, 
                        samp_var1, samp_var2, influence1, influence2, threshold, 
                        sign, fold):
    """     
    INPUTS
    var1_index: integer of bacteria (in bact_names) to be evaluated
    var2_index: integer of metabolite (in meta_names) to be evaluated
    n_samp:     sample size
    n_var1:     number of var1
    n_var2:     number of var2
    samp_var1:  np array where row i col j indicates level of bact j 
                for sample i
    samp_var2:  np array where row i col j indicates level of meta j 
                for sample i
    influence1: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    influence2: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    threshold:  float of level of significance testing (after MC)
    sign:       original sign of correlation to check against following 
                re-evaluation
    fold:       boolean that determines whether you require the new P value to 
                be 100x greater to be labeled cutie

    OUTPUTS
    reverse:  array where index i is 1 if the correlation changes sign upon 
              removing sample i 
    exceeds:  array where index i is 1 if removing that sample causes the 
              correlation to become insignificant in at least 1 different 
              pairwise correlations
    corrs:    1D array of value of correlation strength with index i removed
    p_values: 1D array of value of pvalues with index i removed
    
    FUNCTION
    Takes a given bacteria and metabolite by index and recomputes spearman correlation 
    by removing 1 out of n (sample_size) points from samp_ids. 
    Returns an indicator array where exceeds[i] is 1 if removing that sample causes
    the correlation to become insignificant in at least 1 different pairwise correlations
    """
    exceeds = np.zeros(n_samp)
    reverse = np.zeros(n_samp)
    corrs = np.zeros(n_samp)
    p_values = np.zeros(n_samp)
    var1_values = samp_var1[:,var1_index]
    var2_values = samp_Var2[:,var2_index]
    
    # iteratively delete one sample and recompute statistics
    for sample_index in xrange(n_samp):
        new_var1_values = var1_values[~np.in1d(range(n_var1),sample_index)]
        new_var2_values = var2_values[~np.in1d(range(n_var2),sample_index)]
        if new_var1_values.size <= 3 or new_var2_values.size <= 3:
            p_value = 1
            corr = 0
        else:
            corr, p_value = stats.spearmanr(new_bact, new_meta)
        if fold:
            if (p_value > threshold and p_value > pvalue_matrix[bact_index][meta_index] \
                * 100.0) or np.isnan(p_value):
                exceeds[sample_index] = 1
        elif p_value > threshold or np.isnan(p_value): 
            exceeds[sample_index] = 1
        if np.sign(corr) != sign:
            reverse[sample_index] = 1
        corrs[sample_index] = corr
        p_values[sample_index] = p_value

    return reverse, exceeds, corrs, p_values


def return_indicators(n_var1, n_var2, SLR_initial_sig, SLR_true_sig, resample_k):
    """
    INPUTS
    n_var1:          number of var1
    n_var2:          number of var2
    SLR_initial_sig: list of points/tuples (i,j) if corr between var1, var2 was
                     initially significant
    SLR_true_sig:    dict of list of points/tuples (i,j) if corr between var1, var2 
                     passed significance testing filtering (post-CUtIe) for 
                     key = number of points resampled
    resample_k:      integer of points removed (k = 1, 2, etc.)

    OUTPUT
    indicators: Dict of 2D np array where key is the number of points removed 
                and entry i j corresponds to indicator value for correlation 
                between var1 i and var2 j

    FUNCTION
    Returns an indicator dictionary for the correlations
    """
    indicators = {}
    for i in xrange(resample_k):
        indicators[str(i+1)] = indicator(n_var1, n_var2, SLR_initial_sig,
                                        SLR_true_sig[str(i+1)])

    return indicators

def report_results(n_var1, n_var2, working_dir, label, SLR_initial_sig,
    SLR_true_sig, comb_to_rev, resample_k):
    """
    """

    for i in xrange(resample_k):
        print 'The number of false correlations for ' + str(i+1) + ' is ' + str(
            len(SLR_initial_sig)-len(SLR_true_sig[str(i+1)])) 
        print 'The number of true correlations for ' + str(i+1) + ' is ' + str(
            len(SLR_true_sig[str(i+1)]))

        if comb_to_rev != {}:
            print 'The number of reversed correlations for ' + str(i+1) + ' is ' + str(
                len(comb_to_rev[str(i+1)]))

            # convert dict of pairs to matrix for printing
            n_pairs = len(comb_to_rev[str(i+1)])
            pairs = np.zeros([n_pairs,2])
            for p in xrange(n_pairs):
                pairs[p] = comb_to_rev[str(i+1)][p]

            output.print_matrix(pairs, 
                                working_dir + 'data_processing/' + 'rev_pairs_' + label 
                                + '_resample' + str(i+1) + '.txt', '\t',
                                ['bact_index', 'meta_index'])

    return 


def cookd(var1_index, var2_index, n_samp, n_var1, n_var2, samp_var1, samp_var2, 
    influence1, influence2, threshold, sign, fold ):
    """
    INPUTS
    bact_index:       integer of bacteria (in bact_names) to be evaluated
    meta_index:       integer of metabolite (in meta_names) to be evaluated
    samp_ids:         list of strings of sample ids
    samp_bact_matrix: np array where row i col j indicates level of bact j 
                      for sample i
    samp_meta_matrix: np array where row i col j indicates level of meta j 
                      for sample i
    influence:        object created by sm.OLS to be unpacked
    threshold:        float of level of significance testing (after MC)

    OUTPUTS
    reverse: placeholder output
    exceeds: array where index i is 1 if sample i exceeds threshold

    FUNCTION
    Compute Cook's Distance for a bact, meta pair
    """
    # reverse is 0 because sign never changes
    reverse = np.zeros(n_samp)
    exceeds = np.zeros(n_samp)
    #c is the distance and p is p-value
    (c, p) = influence1.cooks_distance
    for i in xrange(len(c)):
        if c[i] > 1 or np.isnan(c[i]) or c[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds, c, p


def dffits(var1_index, var2_index, n_samp, n_var1, n_var2, samp_var1, samp_var2, 
    influence1, influence2, threshold, sign, fold):
    """
    INPUTS
    var1_index: integer of bacteria (in bact_names) to be evaluated
    var2_index: integer of metabolite (in meta_names) to be evaluated
    n_samp:     sample size
    n_var1:     number of var1
    n_var2:     number of var2
    samp_var1:  np array where row i col j indicates level of bact j 
                for sample i
    samp_var2:  np array where row i col j indicates level of meta j 
                for sample i
    influence1: object created by sm.OLS
    influence2: object created by sm.OLS, placeholder argument
    threshold:  float of level of significance testing (after MC)
    sign:       original sign of correlation to check against following 
                re-evaluation
    fold:       boolean that determines whether you require the new P value to 
                be 100x greater to be labeled cutie

    OUTPUTS
    reverse:  array where index i is 1 if the correlation changes sign upon 
              removing sample i 
    exceeds:  array where index i is 1 if removing that sample causes the 
              correlation to become insignificant in at least 1 different 
              pairwise correlations
    dffits_:  1D array of value of dffits strength with index i removed

    FUNCTION
    Compute DFFITS for a bact, meta pair
    """
    reverse = np.zeros(n_samp)
    exceeds = np.zeros(n_samp)
    dffits_, dffits_threshold = influence1.dffits
    for i in xrange(n_samp):
        if dffits_[i] > dffits_threshold or dffits_[i] < -dffits_threshold or \
        np.isnan(dffits_[i]) or dffits_[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds, dffits_, [dffits_threshold] * n_samp


def dsr(var1_index, var2_index, n_samp, n_var1, n_var2, samp_var1, samp_var2, 
    influence1, influence2, threshold, sign, fold):
    """
    INPUTS
    var1_index: integer of bacteria (in bact_names) to be evaluated
    var2_index: integer of metabolite (in meta_names) to be evaluated
    n_samp:     sample size
    n_var1:     number of var1
    n_var2:     number of var2
    samp_var1:  np array where row i col j indicates level of bact j 
                for sample i
    samp_var2:  np array where row i col j indicates level of meta j 
                for sample i
    influence1: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    influence2: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    threshold:  float of level of significance testing (after MC)
    sign:       original sign of correlation to check against following 
                re-evaluation
    fold:       boolean that determines whether you require the new P value to 
                be 100x greater to be labeled cutie

    OUTPUTS
    reverse:  array where index i is 1 if the correlation changes sign upon 
              removing sample i 
    exceeds:  array where index i is 1 if removing that sample causes the 
              correlation to become insignificant in at least 1 different 
              pairwise correlations
    dsr_:    1D array of value of DSR strength with index i removed
    
    FUNCTION
    Compute Deleted Studentized Residual for a var1, var2 pair
    """
    reverse = np.zeros(n_samp)
    exceeds = np.zeros(n_samp)
    dsr_ = influence1.resid_studentized_external 
    # self.results.resid / sigma / np.sqrt(1 - hii) = influence.resid_studentized_external
    for i in xrange(n_samp):
        if dsr_[i] < -2 or dsr_[i] > 2 or np.isnan(dsr_[i]) or dsr_[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds, dsr_, dsr_


def double_lev(var1_index, var2_index, n_samp, n_var1, n_var2, samp_var1, 
    samp_var2, influence1, influence2, threshold, sign, fold):
    """
    INPUTS
    var1_index: integer of bacteria (in bact_names) to be evaluated
    var2_index: integer of metabolite (in meta_names) to be evaluated
    n_samp:     sample size
    n_var1:     number of var1
    n_var2:     number of var2
    samp_var1:  np array where row i col j indicates level of bact j 
                for sample i
    samp_var2:  np array where row i col j indicates level of meta j 
                for sample i
    influence1: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    influence2: object created by sm.OLS, not relevant to Pearson/Spearman 
                but needed as a placeholder argument
    threshold:  float of level of significance testing (after MC)
    sign:       original sign of correlation to check against following 
                re-evaluation
    fold:       boolean that determines whether you require the new P value to 
                be 100x greater to be labeled cutie

    OUTPUTS
    reverse:      array where index i is 1 if the correlation changes sign upon 
                  removing sample i 
    exceeds:      array where index i is 1 if removing that sample causes the 
                  correlation to become insignificant in at least 1 different 
                  pairwise correlations
    hat_matrix_x: 1D array of value of hat_matrix_x for sample i (hii)
    hat_matrix_y: 1D array of value of hat_matrix_y for sample i (hii)
    
    FUNCTION
    Compute Deleted Studentized Residual for a var1, var2 pair
    """
    # reverse is 0 because sign never changes
    reverse = np.zeros(n_samp)
    exceeds = np.zeros(n_samp)
    #c is the distance and p is p-value
    hat_matrix_x = influence1.hat_matrix_diag
    hat_matrix_y = influence2.hat_matrix_diag
    x_thresh = 3.0*(2.0/float(n_samp))
    y_thresh = 3.0*(2.0/float(n_samp))
    for i in xrange(n_samp):
        if hat_matrix_x[i] > x_thresh and hat_matrix_y[i] > y_thresh:
            exceeds[i] = 1

    return reverse, exceeds, hat_matrix_x, hat_matrix_y

def update_cutie(n_samp, n_var1, n_var2, samp_var1, samp_var2, pvalues, corrs,
    infln_metrics, infln_mapping, threshold = 0.05, paired = False, fold = False):
    """ 
    INPUTS
    n_samp:        intger number of samples
    n_var1:        number of var1
    n_var2:        number of var2
    samp_var1:     np array where row i col j indicates level of var1 j 
                   for sample i
    samp_var2:     np array where row i col j indicates level of var2 j 
                   for sample i
    pvalues:       np array of pvalues where entry i,j corresponds to the 
                   initial pvalue of the correlation between var1 i and var2 j
    corrs:         np array of corrs where entry i,j corresponds to the 
                   initial pvalue of the correlation between var1 i and var2 j
    infln_metrics: list of strings of influential point metrics to use
    infln_mapping: dict mapping string name of metric to the function
    threshold:     float, significance level used for testing (default 0.05)
    
    OUTPUTS
    initial_sig: list of points i,j where the corr(i,j) is significant at 
                 threshold level
    true_sig:    dict where key is a statistic in infln_metrics and the element is
                 a list of i,j points where given metric between var1 i and 
                 var2 j is still sig
    infln_dict:  dictionary of 3D matrices; each with dimension n_samp x n_var1 
                 x n_var2 where entry (k, i, j) refers to the value of a sample 
                 k in correlation var1, var2 = i, j and the key is the metric
    
    FUNCTION
    For all var1, var2 pairs, recomputes pvalues by dropping 1 different 
    observation at a time. Returns a list of var1, var2 points that were 
    initially significant (initial_sig), as well as the subset that remains 
    significant (true_sig).
    """
    # create lists of points
    initial_sig = []
    true_sig = {}
    infln_dict = {}
    corrs_dict = {}
    thresholds_dict = {}

    # 6 for the len of point originally, + 3* number of metrics
    # FYI point = var1, var2, s, samp_var1[s][var1], samp_var2[s][var2], pvalues[var1][var2]
    # 3 -> indicator, cutoff, strength
    n_points = n_samp * n_var1 * n_var2 * len(infln_metrics)
    headers = []
    # due to memory error in python
    all_points = []
    #all_points = np.zeros([n_points, 6 + len(infln_metrics) * 3])
    headers = ['var1_index','var2_index','sample_number','var1_value','var2_value','initial_sig']
    for metric in infln_metrics:
        headers.append(metric + '_indicator')
        headers.append(metric + '_cutoff')
        headers.append(metric + '_strength')

    for metric in infln_metrics:
    # default populate a matrix of 0s, 0 is not significant for the metric, 1 is
        infln_dict[metric] = np.zeros([n_samp, n_var1, n_var2])
        corrs_dict[metric] = np.zeros([n_samp, n_var1, n_var2])
        thresholds_dict[metric] = np.zeros([n_samp, n_var1, n_var2])
        true_sig[metric] = []

    row = 0
    # for each bact, meta pair
    for var1 in xrange(n_var1): 
        for var2 in xrange(n_var2): 
            pair = (var1,var2)
            FP_indicator = 0
            # if variables are paired i.e. the same, then don't compute corr(i,i)
            if pvalues[var1][var2] < threshold and not (paired & (var1 == var2)):
                initial_sig.append(pair)
                FP_indicator = -1
            # set data matrices of x, y
            x_old = samp_var1[:,var1]
            y_old = samp_var2[:,var2]
            # add constant for constant term in regression
            x = sm.add_constant(x_old)
            y = sm.add_constant(y_old)
            # compute models with x and y as independent vars, respectively
            model1 = sm.OLS(y_old,x)
            fitted1 = model1.fit()
            influence1 = fitted1.get_influence()
            model2 = sm.OLS(x_old,y)
            fitted2 = model2.fit()
            influence2 = fitted2.get_influence()   

            # retrieve sign of correlation        
            sign = np.sign(corrs[var1][var2])
            metric_corrs = {}
            metric_thresholds = {}
            metric_exceeds = {}

            # perform cutie resampling for each statistic
            for m in xrange(len(infln_metrics)):
                metric = infln_metrics[m]
                metric_index = headers.index(metric + '_indicator')
                reverse, exceeds, corr_values, thresholds = infln_mapping[metric](
                                var1, var2, n_samp, n_var1, n_var2, 
                                samp_var1, samp_var2, influence1, influence2, 
                                threshold, sign, fold)
                metric_corrs[metric] = corr_values
                metric_thresholds[metric] = thresholds
                metric_exceeds[metric] = exceeds

                # populate dictionary of correlations, thresholds, and indicators
                for i in xrange(n_samp):
                    infln_dict[metric][i][var1][var2] = metric_exceeds[metric][i] 
                    corrs_dict[metric][i][var1][var2] = metric_corrs[metric][i]
                    # thresholds is p-value for cutie pc or sc
                    thresholds_dict[metric][i][var1][var2]  = metric_thresholds[metric][i]

                # add correlation to TP if passed all thresholds
                if pvalues[var1][var2] < threshold and not (paired & (var1 == var2)):
                    if exceeds.sum() == 0:
                        true_sig[metric].append(pair)
                        FP_indicator = 1
            
            # subesquent chunk only works for small enough datasets, not on minerva
            '''
            for i in xrange(len(samp_ids)):
                all_points[row][0] = bact
                all_points[row][1] = meta
                all_points[row][2] = i
                all_points[row][3] = samp_bact_matrix[i][bact]
                all_points[row][4] = samp_meta_matrix[i][meta]
                all_points[row][5] = FP_indicator
                    
                for metric in infln_metrics:
                    metric_index = headers.index(metric + '_indicator')
                    # cutoff
                    all_points[row][metric_index + 1] = metric_thresholds[metric][i]
                    # strength
                    all_points[row][metric_index + 2] = metric_corrs[metric][i]
                    # indicators
                    if pvalues[bact][meta] < threshold:
                        if metric_exceeds[metric][i] == 0:
                            all_points[row][metric_index] = 1
                        else:
                            all_points[row][metric_index] = -1
                    else:
                        all_points[row][metric_index] = 0

                row += 1
            '''
                
    return initial_sig, true_sig, infln_dict, corrs_dict, thresholds_dict, all_points, headers

def pointwise_comparison(n_samp, n_var1, n_var2, samp_var1, samp_var2, pvalues, corrs, 
                        working_dir, n_corr, threshold, point_compare, corr_compare, 
                        statistic, paired = False, fold = False):
    """
    """
    if statistic == 'kpc':
        infln_metrics = ['cookd', 'dffits','dsr']#['cutie_1pc', 'cookd', 'dffits', 'double_lev']#'dsr']#,'double_lev']
        infln_mapping = {
                        #'cutie_1pc': resample1_cutie_pc,
                        'cookd': cookd,
                        'dffits': dffits,
                        'dsr': dsr,
                        #'double_lev': double_lev 
                        }
    elif statistic == 'ksc':
        infln_metrics = ['cookd', 'dffits','dsr']#['cutie_1sc', 'cookd', 'dffits','dsr']#,'double_lev']
        infln_mapping = {
                        #'cutie_1sc': resample1_cutie_sc,
                        'cookd': cookd,
                        'dffits': dffits,
                        'dsr': dsr,
                        #'double_lev': double_lev
                        }

    initial_sig, true_sig, infln_dict, corrs_dict, thresholds_dict, all_points, \
        headers = update_cutie(n_samp, n_var1, n_var2, samp_var1, samp_var2, 
            pvalues, corrs, infln_metrics, infln_mapping, threshold, paired, fold)

    #output.print_matrix(all_points, working_dir + 'data_processing/' + 'all_points_R_matrix_.txt', '\t',headers)
    
    # create list of all possible combinations of influence metrics
    # ex. combs = [['cutie'],['cookd'],['cutie','cookd']]
    combs = []
    for i in xrange(1, len(infln_metrics)+1):
        els = [list(x) for x in itertools.combinations(infln_metrics, i)]
        combs.extend(els)

    # if evaluating each point
    if point_compare:
        point_dict = defaultdict(list)
        for pair in initial_sig:
            var1, var2 = pair
            for s in xrange(len(samp_ids)):
                for comb in combs: # [1], [2, 3], etc.
                    infln = True
                    for c in comb: # each individual key     
                        # 0 means the correlation was never flagged by that influence metric
                        if infln_dict[c][s][var1][var2] == 0:
                            infln = False

                    if infln is True:
                        point = [var1, var2, s, samp_var1[s][var1], \
                            samp_var2[s][var2], pvalues[var1][var2]]
                        for metric in infln_metrics:
                            if infln_dict[metric][s][var1][var2] != 0:
                                exceeded = 1
                            else:
                                exceeded = 0
                            cutoff = thresholds_dict[metric][s][var1][var2]
                            corr_str = corrs_dict[metric][s][var1][var2]
                            point.append(exceeded)
                            point.append(cutoff)
                            point.append(corr_str)
                        point_dict[str(comb)].append(point)

        for comb in combs:
            print 'The amount of influential points in set ' + str(comb) + ' is ' + str(len(point_dict[str(comb)]))
            points = point_dict[str(comb)]
            n_points = len(points)
            # 6 for the len of point originally, + 3* number of metrics
            # FYI point = var1, var2, s, samp_var1[s][var1], samp_var2[s][var2], pvalues[var1][var2]
            point_matrix = np.zeros([n_points, 6+ len(comb) * 3])
            #for p in xrange(n_points):
            #    point_matrix[p] = points[p]
            #headers = ['var1_index','var2_index','sample_number','var1_value','var2_value','pvalue']
            #for c in comb:
            #    headers.append(str(c) + '_indicator')
            #    headers.append(str(c) + '_cutoff')
            #    headers.append(str(c) + '_strength')
            #output.print_matrix(point_matrix, working_dir + 'data_processing/' + 'points_R_matrix' + label + '_' + str(comb) + '.txt', headers)
    
    if corr_compare:
        corr_dict = {}
        for comb in combs:
            corr_dict[str(comb)] = []

        for point in initial_sig:
            var1, var2 = point
            for comb in combs: # [1], [2, 3], etc.
                FP = True
                for c in comb: # each individual key    
                    if sum(infln_dict[c],0)[var1][var2] == 0:
                        FP = False
                        break
                if FP is True:
                    corr_dict[str(comb)].append(point)
                    
        for comb in combs:
            print 'The amount of FP in set ' + str(comb) + ' is ' + str(len(corr_dict[str(comb)]))
            
        n_var1 = np.size(samp_var1,1)
        n_var2 = np.size(samp_var2,1)
        headers = ['var1','var2']
        for metric in infln_metrics:
            headers.append(metric)
        pair_matrix = np.zeros([n_var1 * n_var2 ,len(headers)])
        row = 0
        for var1 in xrange(n_var1):
            for var2 in xrange(n_var2): 
                line = [var1, var2]
                for metric in infln_metrics:
                    point = (var1, var2)
                    if point in initial_sig:
                        if sum(infln_dict[metric],0)[var1][var2] == 0:
                            line.append(1)
                        else:
                            line.append(-1)
                    else:
                        line.append(0)
                pair_matrix[row] = line
                row += 1
        output.print_matrix(pair_matrix, working_dir + 'data_processing/all_pairs.txt', '\t', headers)

        print 'Printing JSON matrix...'
        output.print_json_matrix(n_var1, n_var2, n_corr, infln_metrics, infln_mapping,
                                infln_dict, initial_sig, working_dir,  
                                paired = False)

    # report results
    for metric in infln_metrics:
        metric_true_sig = true_sig[metric]
        print 'The number of false correlations according to ' + metric + \
            ' is ' + str(len(initial_sig)-len(metric_true_sig)) 
        print 'The number of true correlations according to ' + metric + \
            ' is ' + str(len(metric_true_sig))

    return






