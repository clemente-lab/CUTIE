#!/usr/bin/env python
from __future__ import division
    
import os
import numpy as np
from scipy import stats

def initial_stats_SLR(
        samp_ids, 
        samp_bact_matrix, 
        samp_meta_matrix, 
        functions = ['stats.linregress', 'stats.spearmanr'],
        mapf = {'stats.linregress': stats.linregress, 
                'stats.spearmanr': stats.spearmanr},
        f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','pvalue','r2'],
               'stats.spearmanr':
                   ['scorr','pvalue']}
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
                TO DO: For calculations of influence, the data is stored as the 
                measure of influence for bact row i, meta col j, and point
                or sample stack k.
     
    FUNCTION
    Function that computes an initial set of statistics per the specified 
    functions. Returns a dict where the key is a statistical function and  
    the element is an initial matrix with dimensions 3 x num_bact x num_meta,
    corresponding to the pearson corr, p-value, and R2 value from simple linear 
    regression (SLR) between each metabolite and bacteria. 
    
    
    EXAMPLE
    functions = ['stats.linregress', 'stats.spearmanr']
    mapf = {'stats.linregress': stats.linregress,
            'stats.spearmanr': stats.spearmanr}
    f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','pvalue','r2'],
               'stats.spearmanr':
                   ['scorr','pvalue']}
    stat_dict = initial_stats_SLR(
                    samp_ids, 
                    samp_bact_matrix, 
                    samp_meta_matrix)
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

def print_stats_SLR(
        stat_dict,
        working_dir,
        label,
        functions = ['stats.linregress', 'stats.spearmanr'],
        mapf = {'stats.linregress': stats.linregress,
            'stats.spearmanr': stats.spearmanr},
        f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','pvalue','r2'],
               'stats.spearmanr':
                   ['scorr','pvalue']}
        ):
    """ 
    INPUTS   
    stat_dict:   dict that maps string of function to output matrix           
    output_file: string of directory to be saved
    label:       string of label (e.g. 'L6')
    functions:   list of strings of function names 
    mapf:        dict that maps function name to function object
    f_stats:     dict that maps function name to list of output strings
     
    FUNCTION
    Function that prints files for each pertinent statistic (in f_stats)
    for each function in functions corresponding to each bact, meta pair.     
    
    EXAMPLE
    functions = ['stats.linregress', 'stats.spearmanr']
    mapf = {'stats.linregress': stats.linregress,
            'stats.spearmanr': stats.spearmanr}
    f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','pvalue','r2'],
               'stats.spearmanr':
                   ['scorr','pvalue']}
    print_stats_SLR(stat_dict,
                    working_dir,
                    label = 'L6')
    """
    # functions: list of function objects to be called
    # function_stats: dict where key is function and elements are strings
    #     of relevant statistics corresponding to output of function
    
    stat_files = {}
    
    num_bact = np.size(stat_dict[stat_dict.keys()[0]], 1)
    num_meta = np.size(stat_dict[stat_dict.keys()[0]], 2)
    
    # retrieve relevant stats
    for f in functions: # e.g. f = 'stats.linregress'
        rel_stats = f_stats[f] # e.g. ['b1', 'b0', 'pcorr','pvalue','r2']
        num_rel = len(rel_stats) # e.g. num_rel = 5
        # initialize dict of file objects for each relevant statistic
        stat_files[f] = {}
        # retrieve 3D matrix from stat_dict
        matrix = stat_dict[f]
        # create a new file per statistic per function
        for r in xrange(num_rel):
            fname = working_dir + 'data_processing/initial_SLR_' 
            fname = fname + rel_stats[r] + '_' + label + '.txt'
            if os.path.isfile(fname) is True:
                os.remove(fname)
            stat_files[f][rel_stats[r]] = open(fname,'w')
        
        # for each bact, meta pair
        for b in xrange(num_bact):
            for m in xrange(num_meta):
                # retrieve the value pertaining to the statistic of interest
                for r in xrange(num_rel):
                    value = matrix[r][b][m]
                    stat_files[f][rel_stats[r]].write(str(value) + '\t')
            # print a new line after every bacteria
            for r in xrange(num_rel):
                stat_files[f][rel_stats[r]].write('\n')
        # close the files after all bacteria are done
        for r in xrange(num_rel):
            stat_files[f][rel_stats[r]].close()
    return


def resample_SLR(bact_index, meta_index, 
                 samp_ids, samp_bact_matrix, 
                 samp_meta_matrix):
    """ 
    INPUTS
    bact_index:       integer of bacteria (in bact_names) to be evaluated
    meta_index:       integer of metabolite (in meta_names) to be evaluated
    samp_ids:         list of strings of sample ids
    samp_bact_matrix: np array where row i col j indicates level of bact j 
                      for sample i
    samp_meta_matrix: np array where row i col j indicates level of meta j 
                      for sample i
    
    OUTPUTS
    pvalue: array where index i corresponds to the recomputed p-value of the
            correlation between the bact, meta pair with sample i removed
    
    TODO
    updated_stats: dict where key is a function and element is a 2D array where
                   row i col j corresponds to the recomputed statistic when sample
                   i is removed and statistic j is recomputed
                   For key = 'stats.linregress', the statistics are:
                   ['b1', 'b0', 'pcorr','pvalue','r2']
                   For key = 'stats.spearmanr', the statistics are:
                   ['scorr','pvalue']
    
    FUNCTION
    Takes a given bacteria and metabolite by index and recomputes correlation 
    by removing 1 out of n (sample_size) points from subj_id_list. 
    Returns an np.array of pvalues where pvalue[i] corresponds to the pvalue of 
    the correlation after removing the sample of samp_ids[i]
    
    EXAMPLE
    pvalues = resample_SLR(bact, 
                           meta, 
                           samp_ids, 
                           samp_bact_matrix, 
                           samp_meta_matrix)
    """
    
    sample_size = len(samp_ids)
    pvalues = np.zeros(sample_size)
    bact = samp_bact_matrix[:,bact_index]
    meta = samp_meta_matrix[:,meta_index]
    
    # iteratively delete one sample and recompute statistics
    for sample_index in xrange(sample_size):
        new_bact = bact[~np.in1d(range(len(bact)),sample_index)]
        new_meta = meta[~np.in1d(range(len(meta)),sample_index)]
        slope, intercept, r_value, p_value, std_err = stats.linregress(new_bact,
                                                                       new_meta)
        pvalues[sample_index] = p_value
    return pvalues

def update_stats_SLR(
        samp_ids, 
        samp_bact_matrix,
        samp_meta_matrix, 
        pvalue_matrix,
        alpha = 0.05
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
    alpha:            float, significance level used for testing (default 0.05)
    
    OUTPUTS
    initial_sig: list of i,j points where corr of bact i and meta j is initially sig
    true_sig:    list of i,j points where corr of bact i and meta j remains sig
    
    FUNCTION
    For all bact, meta pairs, recomputes pvalues by dropping 1 different 
    observation at a time. Returns a list of bact, meta points that were 
    initially significant (initial_sig), as well as the subset that remains 
    significant (true_sig).
    
    EXAMPLE
    SLR_initial_sig, SLR_true_sig = update_stats_SLR(samp_ids, 
                                                     samp_bact_matrix,
                                                     samp_meta_matrix, 
                                                     pvalue_matrix)
    """
    # pvalue_matrix MUST correspond with correlation_matrix entries
    n_meta = np.size(samp_meta_matrix,1)
    n_bact = np.size(samp_bact_matrix,1)
    n_subj = len(samp_ids)

    # multiple comparisons threshold
    threshold = alpha / (n_meta * n_bact)
    
    # create lists of points
    initial_sig = []
    true_sig = []
    
    # for each bact, meta pair
    for bact in xrange(n_bact): 
        for meta in xrange(n_meta): 
            point = (bact,meta)
            if pvalue_matrix[bact][meta] < threshold:
                initial_sig.append(point)
                pvalues = resample_SLR(bact, 
                                       meta, 
                                       samp_ids, 
                                       samp_bact_matrix, 
                                       samp_meta_matrix)
                # count number of entries above threshold 
                highp = pvalues[np.where(pvalues > threshold)]
                # highp will populate if pvalues exceed threshold, 
                # otherwise sums to 0
                if highp.sum() == 0:
                    true_sig.append(point)

    return initial_sig, true_sig