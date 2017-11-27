#!/usr/bin/env python
from __future__ import division
    
import os
import math
import itertools    
import numpy as np
from scipy import stats
import statsmodels.api as sm


def initial_stats_SLR(
        samp_ids, 
        samp_bact_matrix, 
        samp_meta_matrix, 
        functions,
        mapf,
        f_stats
        ):
    """ 
    INPUTS
    samp_ids:         list of strings of sample IDs
    samp_bact_matrix: np array where row i col j corresponds to level of
                      bact j in sample i
    samp_meta_matrix: np array where row i col j corresponds to level of
                      meta j in sample i                  
    functions:        list of strings of function names 
    mapf:             dict that maps function name to function object
    f_stats:          dict that maps function name to list of output strings
    
    OUTPUTS
    statistics: list of dict where each dict corresponds to each element 
                in function in the order they are passed and where each 
                key corresponds to a particular statistic calculated by 
                each function in functions and the element is an array 
                where row i col j corresponds to the value of a given 
                statistic to the values for bact row i and meta col j.
     
    FUNCTION
    Function that computes an initial set of statistics per the specified 
    functions. Returns a dict where the key is a statistical function and  
    the element is an initial matrix with dimensions num_statistics x num_bact x 
    num_meta, corresponding to the relevant statistics from simple linear 
    regression (SLR) between each metabolite and bacteria. 
    
    
    EXAMPLE
    functions = ['stats.linregress', 'stats.spearmanr']
    mapf = {
        'stats.linregress': stats.linregress,
        'stats.spearmanr': stats.spearmanr
        }
    f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','pvalue','stderr'],
               'stats.spearmanr':
                   ['scorr','pvalue']}
    stat_dict = initial_stats_SLR(
                    samp_ids, 
                    samp_bact_matrix, 
                    samp_meta_matrix,
                    functions,
                    mapf,
                    f_stats)
    """

    stat_dict = {}
    
    num_bact = np.size(samp_bact_matrix, 1)
    num_meta = np.size(samp_meta_matrix, 1)
    
    # retrieve relevant stats and create dictionary entry, 3D array
    for f in functions:
        rel_stats = f_stats[f]
        stat_dict[f] = np.zeros((len(rel_stats), 
                                 num_bact, 
                                 num_meta))

    # subset the data matrices into the cols needed
    for b in xrange(num_bact):
        bact = samp_bact_matrix[:,b]
        for m in xrange(num_meta):
            meta = samp_meta_matrix[:,m] 
            for f in functions:
                # values is a list of the relevant_stats in order
                values = mapf[f](bact, meta)
                for s in xrange(len(values)):
                    stat_dict[f][s][b][m] = values[s] 
    
    return stat_dict 

def multi_zeros(samp_bact_matrix):
    """
    INPUTS
    samp_bact_matrix: 2D array where each entry in row i col j refers to relative 
                      abundance of bacteria j in sample i

    OUTPUTS
    samp_bact_mr:     2D corrected matrix (0's replaced with threshold)     
    samp_bact_clr:    2D centered log ratio matrix, each row of mr divided by its geometric mean 
    samp_bact_lclr:   2D log of CLR matrix, log of each row
    samp_bact_varlog: 1D variance of lclr matrix, element j refers to variance of col j
    correction:       threshold used for correction (currently min(samp_bact_matrix / 2))
    n_zero:           number of 0's detected in the original samp_bact_matrix

    FUNCTION
    Eliminates 0's from a matrix and replaces it with a multiplicative threshold
    correction, using the smallest value divided by 2 as the replacement.
    """
    n_samp = np.size(samp_bact_matrix,0)
    n_bact = np.size(samp_bact_matrix,1)

    # find the minimum value
    min_bact = min(samp_bact_matrix[np.nonzero(samp_bact_matrix)])
    samp_bact_mr = np.copy(samp_bact_matrix)
    # set threshold as minimum value / 2 and replace all 0's
    correction = min_bact / 2.0
    samp_bact_mr[samp_bact_mr == 0] = correction ** 2

    # determine number of 0's in the entire matrix
    n_zero = len(np.where(samp_bact_matrix == 0)[0])

    # correct non-zero values
    for i in xrange(n_samp):
        nrow_zero = len(np.where(samp_bact_matrix[i] == 0)[0])
        for j in xrange(n_bact):
            if samp_bact_mr[i][j] != 0:
                samp_bact_mr[i][j] = samp_bact_mr[i][j] * (1 - nrow_zero * correction)
    

    # create array of geometric means for log clr correction
    samp_bact_gm = np.zeros(n_samp)
    for i in xrange(n_samp):
        samp_bact_gm[i] = math.exp(sum(np.log(samp_bact_mr[i])) / float(n_bact))

    # create log clr correction
    samp_bact_clr = samp_bact_mr / samp_bact_gm[:,None]
    samp_bact_lclr = np.log(samp_bact_clr)

    # create array of variances
    samp_bact_varlog = np.zeros(n_bact)
    for i in xrange(n_bact):
        samp_bact_varlog[i] = np.var(samp_bact_lclr[:,i])
    
    return samp_bact_mr, samp_bact_clr, samp_bact_lclr, samp_bact_varlog, correction, n_zero


def initial_stats_prop(samp_bact_clr, samp_bact_varlog):
    """
    INPUTS
    samp_bact_clr:    2D centered log ratio matrix, each row of mr divided by its geometric mean 
    samp_bact_varlog: 1D variance of lclr matrix, element j refers to variance of col j

    OUTPUTS
    prop: 2D matrix where row i col j corresponds to the proportionality statistic between bact i and j

    FUNCTION
    Computes proportionality statistics as defined in Lovell et al. (2015)
    """
    n_samp = np.size(samp_bact_clr,0)
    n_bact = np.size(samp_bact_clr,1)

    # compute phi matrix
    prop = np.zeros([n_bact,n_bact])
    for i in xrange(n_bact):
        for j in xrange(i):
            prop[i][j] = np.var(np.log(np.divide(samp_bact_clr[:,i], samp_bact_clr[:,j]))) / samp_bact_varlog[i]
            
            # alternative full definition 
            # beta[i]j] = samp_bact_var[j] / samp_bact_var[i] 
            # r[i][j] = 2.0 * np.corrcoef(samp_bact_clr[:,j] , samp_bact_clr[:,i])[1][0] / np.sqrt(samp_bact_var[j] * samp_bact_var[i]) 
            # phi[i][j] = 1.0 + beta[i][j] ** 2 - 2 * beta[i][j] * np.absolute(r[i][j] 

    return prop

def resamplek_cutie_prop(bact1_index, 
                         bact2_index, 
                         samp_bact_clr, 
                         prop_threshold = 0.05,
                         k = 1):
    """
    INPUTS
    bact1_index:    row index of bacteria to test 
    bact2_index:    col index of bacteria to test
    samp_bact_clr:  2D centered log ratio matrix, each row of mr divided by its geometric mean 
    prop_threshold: one-sided threshold cutoff for proportionality, default 0.05
    k:              number of resamplings CUtIe performs

    OUTPUTS
    exceeds: array where index i is k if removing that sample causes the 
             correlation to become insignificant in k different pairwise correlations

    FUNCTION
    Computes resampling and evaluation of significance of proportionality
    """
    sample_size = np.size(samp_bact_clr,0)
    exceeds = np.zeros(sample_size)
    bact1 = samp_bact_clr[:,bact1_index]
    bact2 = samp_bact_clr[:,bact2_index]
    
    # iteratively delete two samples and recompute statistics
    combs = [list(x) for x in itertools.combinations(xrange(sample_size), k)]
    for indices in combs:
        new_bact1 = bact1[~np.in1d(range(len(bact1)),indices)]
        new_bact2 = bact2[~np.in1d(range(len(bact2)),indices)]

        prop = np.var(np.log(np.divide(new_bact1, new_bact2))) / np.var(np.log(new_bact1))

        if prop > prop_threshold or prop == 0.0: 
            for i in indices:
                exceeds[i] += 1

    return exceeds

def initial_stats_conc(samp_bact_lclr, samp_bact_varlog):
    """
    INPUTS
    samp_bact_clr:    2D centered log ratio matrix, each row of mr divided by its geometric mean 
    samp_bact_varlog: 1D variance of lclr matrix, element j refers to variance of col j

    OUTPUTS
    concordance: 2D matrix where row i col j corresponds to the proportionality statistic 
                 between bact i and j

    FUNCTION
    Computes concordance statistics as defined in Zheng (2000)
    """
    n_samp = np.size(samp_bact_lclr,0)
    n_bact = np.size(samp_bact_lclr,1)

    # compute phi matrix
    concordance = np.zeros([n_bact,n_bact])
    for i in xrange(n_bact):
        for j in xrange(i):
            concordance[i][j] = 2.0 * np.cov(samp_bact_lclr[:,i], samp_bact_lclr[:,j])[0][1] / (samp_bact_varlog[i] + samp_bact_varlog[j])

    return concordance

def resamplek_cutie_conc(bact1_index, 
                         bact2_index, 
                         samp_bact_lclr, 
                         conc_threshold,
                         k,
                         sign):
    """     
    INPUTS
    bact1_index:    row index of bacteria to test 
    bact2_index:    col index of bacteria to test
    samp_bact_lclr: 2D log of CLR matrix, log of each row
    conc_threshold: one-sided threshold cutoff for proportionality
    k:              number of resamplings CUtIe performs
    sign:           keeps track of sign as concordance runs from -1 to 1

    OUTPUTS
    exceeds: array where index i is k if removing that sample causes the 
             correlation to become insignificant in k different pairwise correlations

    FUNCTION
    Computes resampling and evaluation of significance of proportionality
    """
    sample_size = np.size(samp_bact_lclr,0)
    exceeds = np.zeros(sample_size)
    reverse = np.zeros(sample_size)
    bact1 = samp_bact_lclr[:,bact1_index]
    bact2 = samp_bact_lclr[:,bact2_index]
    
    # iteratively delete two samples and recompute statistics
    combs = [list(x) for x in itertools.combinations(xrange(sample_size), k)]
    for indices in combs:
        new_bact1 = bact1[~np.in1d(range(len(bact1)),indices)]
        new_bact2 = bact2[~np.in1d(range(len(bact2)),indices)]

        concordance = 2.0 * np.cov(new_bact1, new_bact2)[0][1] / (np.var(new_bact1) + np.var(new_bact2))

        if np.absolute(concordance) < conc_threshold: 
            for i in indices:
                exceeds[i] += 1
        if np.sign(concordance) != sign:
            for i in indices:
                reverse[i] += 1

    return reverse, exceeds


def set_threshold(
    pvalue_matrix,
    alpha,
    mc
    ):

    """
    INPUTS
    pvalue_matrix: 2D np array of pvalues
    alpha:         float of original cutoff
    mc:            form of multiple corrections to use
                    nomc: none
                    bc: bonferroni
                    fwer: family-wise error rate 
                    fdr: false discovery rate
    OUTPUTS
    threshold: float cutoff of pvalues

    FUNCTION
    Performs a multiple comparisons correction on the alpha value
    """
    # determine threshold based on multiple comparisons setting
    if mc == 'nomc':
        threshold = alpha
    elif mc == 'bc':
        threshold = alpha / pvalue_matrix.size
    elif mc == 'fwer':
        threshold = 1.0 - (1.0 - alpha) ** (1/(pvalue_matrix.size))
    elif mc == 'fdr':
        # compute constant in FDR correction
        pvalues = np.sort(pvalue_matrix.flatten())
        k = 0
        j = 0
        '''
        Alternative FDR constant that is more correct 
        cn = 0.0
        for i in xrange(len(pvalues)):
            cn += 1.0/(i+1.0)
        '''
        cn = 1.0
        for k in xrange(len(pvalues)):
            threshold = (float(k+1))/(len(pvalues)) * 0.05/cn
            if pvalues[k] < threshold:
                j = k 
                while pvalues[j] < threshold:
                    j += 1
                    threshold = (float(j+1))/(len(pvalues)) * 0.05/cn
                break
        if j == 0:
            print 'Warning: no p-values below threshold, defaulted to min(p) = ' + str(min(pvalues))
        else:
            threshold = (float(j))/(len(pvalues)) * 0.05/cn

    return threshold


def indicator(n_bact, 
              n_meta,
              initial_sig,
              true_sig):
    """
    INPUTS
    n_bact:      number of bacteria
    n_meta:      number of metabolites
    initial_sig: list of bact,meta points that were significant correlations
                 prior to resampling
    true_sig:    list of bact,meta points that remain significant following
                 resampling

    OUTPUTS
    indicators: array of size (n_bact x n_meta) where each i,j entry is
                0  if bact i, meta j were never significantly correlated (TN)
                -1 if bact i, meta j were falsely correlated (FP)
                1  if bact i, meta j remain correlated following resampling (TP)

    FUNCTION
    Takes in lists of initially significant points and truly significant points
    and returns a matrix indicating TN, FP and TP
    """
    indicators = np.zeros((n_bact,n_meta))
    for point in initial_sig:
        i,j = point
        indicators[i][j] = -1
    for point in true_sig:
        i,j = point
        indicators[i][j] = 1
    return indicators

def resample1_cutie_pc(bact_index, 
                       meta_index, 
                       samp_ids, 
                       samp_bact_matrix, 
                       samp_meta_matrix, 
                       influence,
                       threshold,
                       sign):
    """     
    INPUTS
    bact_index:       integer of bacteria (in bact_names) to be evaluated
    meta_index:       integer of metabolite (in meta_names) to be evaluated
    samp_ids:         list of strings of sample ids
    samp_bact_matrix: np array where row i col j indicates level of bact j 
                      for sample i
    samp_meta_matrix: np array where row i col j indicates level of meta j 
                      for sample i
    influence:        object created by sm.OLS, not relevant to CUtIe but needed as
                      a placeholder argument
    threshold:        float of level of significance testing (after MC)
    sign:             original sign of correlation to check against following re-evaluation
    
    OUTPUTS
    reverse: array where index i is 1 if the correlation changes sign upon removing
             sample i 
    exceeds: array where index i is 1 if removing that sample causes the 
             correlation to become insignificant in at least 1 different pairwise correlations
    
    FUNCTION
    Takes a given bacteria and metabolite by index and recomputes pearson correlation 
    by removing 1 out of n (sample_size) points from samp_ids. 
    Returns an indicator array where exceeds[i] is 1 if removing that sample causes
    the correlation to become insignificant in at least 1 different pairwise correlations
    """
    
    sample_size = len(samp_ids)
    exceeds = np.zeros(sample_size)
    reverse = np.zeros(sample_size)
    bact = samp_bact_matrix[:,bact_index]
    meta = samp_meta_matrix[:,meta_index]
    
    # iteratively delete one sample and recompute statistics
    for sample_index in xrange(sample_size):
        new_bact = bact[~np.in1d(range(len(bact)),sample_index)]
        new_meta = meta[~np.in1d(range(len(meta)),sample_index)]
        slope, intercept, r_value, p_value, std_err = stats.linregress(new_bact,
                                                                       new_meta)
        if p_value > threshold: 
            exceeds[sample_index] = 1
        if np.sign(r_value) != sign:
            reverse[sample_index] = 1

    return reverse, exceeds

def resamplek_cutie_pc(bact_index, 
                       meta_index, 
                       samp_ids, 
                       samp_bact_matrix, 
                       samp_meta_matrix, 
                       threshold,
                       k,
                       sign):
    """     
    INPUTS
    bact_index:       integer of bacteria (in bact_names) to be evaluated
    meta_index:       integer of metabolite (in meta_names) to be evaluated
    samp_ids:         list of strings of sample ids
    samp_bact_matrix: np array where row i col j indicates level of bact j 
                      for sample i
    samp_meta_matrix: np array where row i col j indicates level of meta j 
                      for sample i
    influence:        object created by sm.OLS, not relevant to CUtIe but needed as
                      a placeholder argument
    threshold:        float of level of significance testing (after MC)
    sign:             original sign of correlation to check against following re-evaluation
    
    OUTPUTS
    reverse: array where index i is 1 if the correlation changes sign upon removing
             sample i 
    exceeds: array where index i is k if removing that sample causes the 
             correlation to become insignificant in k different pairwise correlations
    
    FUNCTION
    Takes a given bacteria and metabolite by index and recomputes pearson correlation 
    by removing 1 out of n (sample_size) points from samp_ids. 
    Returns an indicator array where exceeds[i] is 1 if removing that sample causes
    the correlation to become insignificant in k different pairwise correlations
    """
    sample_size = len(samp_ids)
    exceeds = np.zeros(sample_size)
    reverse = np.zeros(sample_size)
    bact = samp_bact_matrix[:,bact_index]
    meta = samp_meta_matrix[:,meta_index]
    
    # iteratively delete two samples and recompute statistics
    combs = [list(x) for x in itertools.combinations(xrange(sample_size), k)]
    for indices in combs:
        new_bact = bact[~np.in1d(range(len(bact)),indices)]
        new_meta = meta[~np.in1d(range(len(meta)),indices)]
        slope, intercept, r_value, p_value, std_err = stats.linregress(new_bact,
                                                                       new_meta)

        if p_value > threshold: 
            for i in indices:
                exceeds[i] += 1
        if np.sign(r_value) != sign:
            for i in indices:
                reverse[i] += 1

    return reverse, exceeds

def resample1_cutie_sc(bact_index, 
                       meta_index, 
                       samp_ids, 
                       samp_bact_matrix, 
                       samp_meta_matrix, 
                       influence,
                       threshold,
                       sign):
    """     
    INPUTS
    bact_index:       integer of bacteria (in bact_names) to be evaluated
    meta_index:       integer of metabolite (in meta_names) to be evaluated
    samp_ids:         list of strings of sample ids
    samp_bact_matrix: np array where row i col j indicates level of bact j 
                      for sample i
    samp_meta_matrix: np array where row i col j indicates level of meta j 
                      for sample i
    influence:        object created by sm.OLS, not relevant to CUtIe but needed as
                      a placeholder argument
    threshold:        float of level of significance testing (after MC)
    sign:             original sign of correlation to check against following re-evaluation
    
    OUTPUTS
    reverse: array where index i is 1 if the correlation changes sign upon removing
             sample i 
    exceeds: array where index i is k if removing that sample causes the 
             correlation to become insignificant in at least 1 different pairwise correlations
    
    FUNCTION
    Takes a given bacteria and metabolite by index and recomputes spearman correlation 
    by removing 1 out of n (sample_size) points from samp_ids. 
    Returns an indicator array where exceeds[i] is 1 if removing that sample causes
    the correlation to become insignificant in at least 1 different pairwise correlations
    """
    
    sample_size = len(samp_ids)
    exceeds = np.zeros(sample_size)
    reverse = np.zeros(sample_size)
    bact = samp_bact_matrix[:,bact_index]
    meta = samp_meta_matrix[:,meta_index]
    
    # iteratively delete one sample and recompute statistics
    for sample_index in xrange(sample_size):
        new_bact = bact[~np.in1d(range(len(bact)),sample_index)]
        new_meta = meta[~np.in1d(range(len(meta)),sample_index)]
        corr, p_value = stats.spearmanr(new_bact, new_meta)
        if p_value > threshold: 
            exceeds[sample_index] = 1
        if np.sign(corr) != sign:
            reverse[sample_index] = 1
    return reverse, exceeds

def resamplek_cutie_sc(bact_index, 
                       meta_index, 
                       samp_ids, 
                       samp_bact_matrix, 
                       samp_meta_matrix, 
                       threshold,
                       k,
                       sign):
    """     
    INPUTS
    bact_index:       integer of bacteria (in bact_names) to be evaluated
    meta_index:       integer of metabolite (in meta_names) to be evaluated
    samp_ids:         list of strings of sample ids
    samp_bact_matrix: np array where row i col j indicates level of bact j 
                      for sample i
    samp_meta_matrix: np array where row i col j indicates level of meta j 
                      for sample i
    influence:        object created by sm.OLS, not relevant to CUtIe but needed as
                      a placeholder argument
    threshold:        float of level of significance testing (after MC)
    k:                number of points removed at most
    sign:             original sign of correlation to check against following re-evaluation
    
    OUTPUTS
    reverse: array where index i is 1 if the correlation changes sign upon removing
             sample i 
    exceeds: array where index i is k if removing that sample causes the 
             correlation to become insignificant in k different pairwise correlations
    
    FUNCTION
    Takes a given bacteria and metabolite by index and recomputes spearman correlation 
    by removing 1 out of n (sample_size) points from samp_ids. 
    Returns an indicator array where exceeds[i] is 1 if removing that sample causes
    the correlation to become insignificant in k different pairwise correlations
    """
    sample_size = len(samp_ids)
    exceeds = np.zeros(sample_size)
    reverse = np.zeros(sample_size)
    bact = samp_bact_matrix[:,bact_index]
    meta = samp_meta_matrix[:,meta_index]
    
    # iteratively delete two samples and recompute statistics
    combs = [list(x) for x in itertools.combinations(xrange(sample_size), k)]
    for indices in combs:
        new_bact = bact[~np.in1d(range(len(bact)),indices)]
        new_meta = meta[~np.in1d(range(len(meta)),indices)]
        corr, p_value = stats.spearmanr(new_bact, new_meta)

        if p_value > threshold: # or np.sign(corr) != sign:
            for i in indices:
                exceeds[i] += 1
        if np.sign(corr) != sign:
            for i in indices:
                reverse[i] += 1

    return reverse, exceeds

def cookd(bact_index, meta_index, 
          samp_ids, samp_bact_matrix, 
          samp_meta_matrix, influence,
          threshold):
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
    reverse = np.zeros(len(samp_ids))
    exceeds = np.zeros(len(samp_ids))
    #c is the distance and p is p-value
    (c, p) = influence.cooks_distance
    for i in xrange(len(c)):
        if c[i] > 1 or np.isnan(c[i]) or c[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds


def dffits(bact_index, meta_index, 
           samp_ids, samp_bact_matrix, 
           samp_meta_matrix, influence,
           threshold):
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
    Compute DFFITS for a bact, meta pair
    """
    reverse = np.zeros(len(samp_ids))
    exceeds = np.zeros(len(samp_ids))
    dffits_, dffits_threshold = influence.dffits
    for i in xrange(len(dffits_)):
        if dffits_[i] > dffits_threshold or np.isnan(dffits_[i]) or dffits_[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds


def dsr(bact_index, meta_index, 
        samp_ids, samp_bact_matrix, 
        samp_meta_matrix, influence,
        threshold):
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
    Compute Deleted Studentized Residual for a bact, meta pair
    """
    reverse = np.zeros(len(samp_ids))
    exceeds = np.zeros(len(samp_ids))
    dsr_ = influence.resid_studentized_external 
    # self.results.resid / sigma / np.sqrt(1 - hii) = influence.resid_studentized_external
    for i in xrange(len(dsr_)):
        if dsr_[i] < -2 or dsr_[i] > 2 or np.isnan(dsr_[i]) or dsr_[i] == 0.0:
            exceeds[i] = 1

    return reverse, exceeds


def update_cutie(
        samp_ids, 
        samp_bact_matrix,
        samp_meta_matrix, 
        pvalue_matrix,
        corr_matrix,
        infln_metrics,
        infln_mapping,
        threshold = 0.05
        ):
    """ 
    INPUTS
    samp_ids:         list of strings ofsample ids
    samp_bact_matrix: np array where row i col j indicates level of bact j 
                      for sample i
    samp_meta_matrix: np array where row i col j indicates level of meta j 
                      for sample i
    pvalue_matrix:    np array of pvalues where entry i,j corresponds to the 
                      initial pvalue of the correlation between bact i and meta j
    infln_metrics:    list of strings of influential point metrics to use
    infln_mapping:    dict mapping string name of metric to the function
    threshold:        float, significance level used for testing (default 0.05)
    
    OUTPUTS
    initial_sig: list of points i,j where the corr(i,j) is significant at threshold level
    true_sig:    dict where key is a statistic in infln_metrics and the element is
                 a list of i,j points where given metric between bact i and meta j is still sig
    infln_dict:  dictionary of 3D matrices; each with dimension n_sample x n_bact x n_meta 
                 where entry (k, i, j) refers to the value of a sample k in correlation 
                 bact, meta = i, j and the key is the metric
    
    FUNCTION
    For all bact, meta pairs, recomputes pvalues by dropping 1 different 
    observation at a time. Returns a list of bact, meta points that were 
    initially significant (initial_sig), as well as the subset that remains 
    significant (true_sig).
    
    EXAMPLE
    infln_metrics = ['cutie','cookd','dffits','dsr']
    infln_mapping = {
                    'cutie': resample_cutie,
                    'cookd': cookd,
                    'dffits': dffits,
                    'dsr': dsr
                    }
    SLR_initial_sig, SLR_true_sig = update_cutie(samp_ids, 
                                                 samp_bact_matrix,
                                                 samp_meta_matrix, 
                                                 pvalue_matrix,
                                                 infln_metrics,
                                                 infln_mapping,
                                                 threshold)
    """
    # pvalue_matrix MUST correspond with correlation_matrix entries
    n_meta = np.size(samp_meta_matrix,1)
    n_bact = np.size(samp_bact_matrix,1)
    n_samp = len(samp_ids)

    # create lists of points
    initial_sig = []
    true_sig = {}
    infln_dict = {}

    for metric in infln_metrics:
    # default populate a matrix of 0s, 0 is not significant for the metric, 1 is
        infln_dict[metric] = np.zeros([n_samp, n_bact, n_meta])
        true_sig[metric] = []

    # for each bact, meta pair
    for bact in xrange(n_bact): 
        for meta in xrange(n_meta): 
            point = (bact,meta)
            if pvalue_matrix[bact][meta] < threshold and pvalue_matrix[bact][meta] != 0.0:
                initial_sig.append(point)
                x = samp_bact_matrix[:,bact]
                y = samp_meta_matrix[:,meta]
                x = sm.add_constant(x)
                model = sm.OLS(y,x)
                fitted = model.fit()
                influence = fitted.get_influence()
                sign = np.sign(corr_matrix[bact][meta])
                for metric in infln_metrics:
                    reverse, exceeds = infln_mapping[metric](
                                           bact, 
                                           meta, 
                                           samp_ids, 
                                           samp_bact_matrix, 
                                           samp_meta_matrix,
                                           influence,
                                           threshold,
                                           sign)
                    for i in xrange(len(exceeds)):
                        infln_dict[metric][i][bact][meta] = exceeds[i] 

                    # sums to 0
                    if exceeds.sum() == 0:
                        true_sig[metric].append(point)

    return initial_sig, true_sig, infln_dict

