#!/usr/bin/env python
from __future__ import division

import re
import sys
import os
import csv
import numpy as np
from itertools import izip
from scipy import stats

def mapping_parse (samp_meta_file, 
                     startcol=17, 
                     endcol=100,
                     delimiter='\t'):
    """
    INPUTS
    samp_meta_file: file object pointing to a table relating samples 
                    to metabolite levels
    startcol:       integer corresponding to the first column containing 
                    metabolite data in samp_meta_file
    endcol:         integer corresponding to the column AFTER the last 
                    column containing metabolite data in samp_meta_file
    
    OUTPUTS
    samp_ids:   list of strings of sample IDs
    meta_names: list of strings of metabolite names 
    samp_meta:  dict where key = samp id (string) and entry is a list of 
                floats containing metabolite levels
    
    FUNCTION
    Reads in a sample and metabolite data file and returns a list of sample ids 
    (samp_ids), a list of metabolites (meta_names), and a dict mapping 
    samp_ids to metabolite levels.
    
    EXAMPLE
    samp_meta_file = open(
        'data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt',
        'r')
    samp_id_list, metabolite_list = samp_meta_parse(
                                        samp_meta_file,
                                        17,99)
    """
    samp_ids = []
    samp_meta = {}
    # generate metabolite list from the 0th line (header)
    # default assumes metabolites are in col 17 to 99
    meta_names = samp_meta_file.readline().split(delimiter)[startcol:endcol]
    # for the remainder of the lines (i.e. the non-header lines)
    for line in samp_meta_file:
        if line != '\n':
            line = line.split('\n')[0]
            line = line.split(delimiter)
            samp_ids.append(line[0]) # line[0] is the sample id
            metabolite_levels = [np.nan if x == '' else float(x) for x in \
                line[startcol:endcol]]
            while len(metabolite_levels) < len(meta_names):
                metabolite_levels.append(np.nan)
            samp_meta[line[0]] = metabolite_levels
    n_meta = len(meta_names)
    n_samp = len(samp_ids)
    print 'The length of mapping_variables is ' + str(n_meta)
    print 'The number of samples is ' + str(n_samp)
    return samp_ids, meta_names, samp_meta, n_meta, n_samp

def otu_parse(samp_bact_file, delimiter = '\t', skip = 1):
    """ 
    INPUTS
    samp_bact_file: file object pointing to an OTU table of bacteria levels
    
    OUTPUTS
    bact_names: list of strings corresponding to bacteria names 
                (in the OTU table)
    samp_bact:  dict where key = sample id (string) and entry is a string
                containing relative abundance of bacteria
    samp_ids:   list of strings of sample IDs

    FUNCTION
    Reads in a sample-bacteria file (OTU table) and returns a list of bacteria 
    names (bact_names) and a dict (samp_bact) mapping sample ids to relative
    abundances.
    
    EXAMPLE
    with open('data/otu_table_MultiO__Status_merged___L6.txt', 'r') as f:
        bact_list_L6 = samp_bact_parse(f)
    """
    
    # create lists (corresponding to smoking and non-smoking files) 
    bact_names = []
    samp_bact = {}
    
    for i in xrange(skip):
        samp_bact_file.readline() # line 0 is 'constructed from biom file' 

    samp_ids = samp_bact_file.readline().rstrip().split(delimiter)
    samp_ids.pop(0) # the 0th entry is a header
    for samp_id in samp_ids:
        samp_bact[samp_id] = []

    for line in samp_bact_file:
        if line is not '': 
            split_line = line.rstrip().split(delimiter)
            # the 0th entry is the name of an OTU
            bact_names.append(split_line[0])
            split_line.pop(0) # pop off OTU
            for b in xrange(len(split_line)):
                samp_bact[samp_ids[b]].append(split_line[b])
        
    n_bact = len(bact_names)
    n_samp = len(samp_ids)

    print 'The length of samp_ids is ' + str(n_samp)
    print 'The length of bact_names is ' + str(n_bact)

    return samp_ids, bact_names, samp_bact, n_bact, n_samp




def parse_input(ftype, fp, startcol, endcol, delimiter, skip):
    """
    """
    # some files like the mapping file won't split on \n but will on \rU
    if ftype == 'map':
        with open(fp,'rU') as f:    
            samp_ids, var_names, samp_to_var, n_var, n_samp = \
                mapping_parse(f, startcol, endcol, delimiter)
   
    elif ftype == 'otu':
        with open(fp, 'rU') as f:
            samp_ids, var_names, samp_to_var, n_var, n_samp = \
                otu_parse(f, delimiter, skip)     

    return samp_ids, var_names, samp_to_var, n_var, n_samp 

def parse_sparcc(sparcc_cov_fp, sparcc_pvalue_fp, delimiter, var_names, n_var):
    """
    """
    sparcc_cov = np.zeros(shape=[n_var,n_var])
    sparcc_pvalues = np.ones(shape=[n_var,n_var])

    # initialize headers and dict
    headers = f1.readline().rstrip().split(delimiter)
    # placeholder index for the header for otus
    indices = [np.nan]
    indices.extend([var_names.index(x) for x in headers[1:]])    

    for line1, line2 in izip(f1, f2):
        split_line1 = line1.rstrip().split(delimiter)
        split_line2 = line2.rstrip().split(delimiter)
        for i in xrange(len(split_line1)):  
            var1_index = indices[row]   
            var2_index = indices[i]           
            sparcc_cov[var1_index][var2_index] = split_line1[i]
            sparcc_pvalues[var1_index][var2_index] = split_line2[i]
        row += 1

    return sparcc_cov, sparcc_pvalues

def dict_to_matrix(samp_dict, samp_ids):
    """ 
    INPUTS
    samp_dict: dict where key = string of sample IDs and element is 
               list of floats corresponding to level of bact or meta
    samp_ids:  list of strings of sample IDs
    
    OUTPUTS
    samp_matrix: np array where each value in row i col j is the level of 
                 bact or meta j corresponding to sample i in the order that
                 the samples are presented in samp_ids
                 
    FUNCTION
    Reads in a dict where the key is the sample id and the subsequent 
    tab-delimited values are either metabolite or bacteria levels and 
    loads it into an np.array where row i and column j correspond 
    to the value of sample i, the rows being sorted in the order that 
    they are inputted per samp_ids
    
    EXAMPLE
    samp_bact_matrix = dict_to_matrix(samp_bact_dict_L6, samp_ids)
    samp_meta_matrix = dict_to_matrix(samp_meta_dict, samp_ids)
    """
    
    # initialize matrix; # rows = # of samp_ids, # cols = # entries per key
    rows = len(samp_ids)
    cols = len(samp_dict[samp_dict.keys()[0]])
    samp_matrix = np.zeros(shape=(rows,cols))    
    
    # populate matrix from the dict
    for r in xrange(rows):
        for c in xrange(cols):
            samp_matrix[r][c] = samp_dict[samp_ids[r]][c]

    # retrieve mean value
    avg_matrix = np.array([np.mean(samp_matrix,0)])
    var_matrix = np.array([np.var(samp_matrix,0)])
    skew_matrix = np.array([[stats.skew(samp_matrix[:,x]) for x in xrange(cols)]])
    return samp_matrix, avg_matrix, var_matrix, skew_matrix


def transpose_csv(f, transposed_fp, skip = 0):
    """
    FUNCTION
    Tranposes .csv file, skipping given number of lines
    """
    # skip lines
    for i in xrange(skip):
        f.readline()
    # retrieve iterable and transpose
    a = izip(*csv.reader(f))
    # write file
    csv.writer(open(transposed_fp, "wb")).writerows(a)
    return

def subset_data(n_samp, transposed_fn, transposed_fp, working_dir):
    """
    n_samp:           int number of samples
    transposed_fn:    string of file name of transposed version of original dataset
                      e.g. otu_transpose_table_small.MSQ34_L6.csv
    tranposed_fp:     string of fp of transposed original dataset
                      e.g. working_dir/otu_transpose_table_small.MSQ34_L6.csv
    working_dir:      working directory
    """
    # create subsetted data files (removing one sample from each by deleting row via sed)
    # row deleted is the prefix of the file e.g.
    # resample_fp = working_dir/0_otu_transpose_table_small.MSQ34_L6.csv
    for k in xrange(n_samp): 
        resample_fp = working_dir + str(k) + '_' + transposed_fn
        if os.path.isfile(resample_fp) == False:
            # sed to delete row
            # sed is 1 indexed, the top row is the header, hence the k + 2
            os.system("sed " + str(k+2)+ "d " + transposed_fp + " > " + \
                resample_fp)

    return

def parse_mine(mine_fp, n_var, var_names, 
               statistics = ['MIC_str','MIC_nonlin','MAS_nonmono','MEV_func',\
                            'MCN_comp','linear_corr'], 
               delimiter = ','):
    """
    INPUTS
    mine_fp:    file path of results from MINE e.g.
                otu_transpose_table_small.MSQ34_L6.csv,allpairs,cv=0.1,B=n^0.6,Results.csv
    n_var:      int number of variables
    var_names:  list of strings of variable names
    statistics: list of statistics that MINE computes
                ['MIC_str','MIC_nonlin','MAS_nonmono','MEV_func','MCN_comp','linear_corr'] 
    delimiter:  ',' MINE uses csv files by default

    OUTPUT
    stat_to_matrix: dictionary where key = statistic (string), entry = matrix 
                    where i,j is the value of statistic for correlation between 
                    var i and j
    FUNCTION
    Parses a MINE output file and stores its statistics for each correlation in 
    a dictionary of matrices

    """
    # skip header (stored if needed later)
    headers = mine_fp.readline().rstrip().split(delimiter)
    
    # prepare dictionary of matrices corresponding to each statistic for each 
    # correlation
    # correlations that are not reported by mine have a default value of 0 in 
    # each category
    stat_to_matrix = {}
    for statistic in statistics:
        stat_to_matrix[statistic] = np.zeros(shape=[n_var,n_var])

    # parse file, header and example line:
    # X var, Y var, MIC (strength), MIC-p^2 (nonlinearity), MAS (non-monotonicity), 
    # MEV (functionality), MCN (complexity),Linear regression (p)
    # otu1,otu2,0.62378645,0,-3.40E+38,2,0.6133625
    for line in mine_fp.readlines():
        split_line = line.rstrip().split(delimiter)
        var1_index = var_names.index(split_line[0]) # X var = otu1
        var2_index = var_names.index(split_line[1]) # Y var = otu2
        for i in xrange(len(statistics)): 
            value = split_line[i + 2] # everything after X and Y var
            # populate both upper and lower diagonals, as MINE only returns i,j 
            # or j,i (is not consistent)
            stat_to_matrix[statistics[i]][var1_index][var2_index] = value
            stat_to_matrix[statistics[i]][var2_index][var1_index] = value
    return stat_to_matrix

def parse_minep(pvalue_fp, delimiter = ',', pskip = 13):
    """
    INTPUTS
    pvalue_fp: table of pvalue-MICstrength relationship provided by MINE
    delimiter: ',' MINE uses csv files by default
    pskip:     number of rows to skip in the pvalue table (various comments)

    OUTPUTS
    MINE_bins:       array where each row has [MIC_str, pvalue, stderr of pvalue]
                     (pvalue corresponds to probability of observing MIC_str as and 
                     more extreme as observed MIC_str)
    pvalues_ordered: sorted list of pvalues from greatest to least used by MINE 
                     to bin

    FUNCTION
    Parses a MINE pvalue table into bins that relate MIC-str to pvalue
    """
    # initialize lists
    MINE_bins = []
    pvalues_ordered = []
    # skip comments
    for i in xrange(pskip):
        pvalue_fp.readline()
    # parse file
    for line in pvalue_fp.readlines():
        # example line: 1.000000,0.000000256,0.000000181
        # corresonding to [MIC_str, pvalue, stderr of pvalue]
        split_line = line.rstrip().split(delimiter)
        # make sure line is valid; last line is 'xla' 
        if len(split_line) > 1:
            row = [float(x) for x in split_line]
            MINE_bins.append(row)
            pvalues_ordered.append(row[0]) # row[0] is the pvalue

    # convert list to array
    MINE_bins = np.array(MINE_bins)

    return MINE_bins, pvalues_ordered 
