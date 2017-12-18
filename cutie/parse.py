#!/usr/bin/env python
from __future__ import division

import re
import sys
import os
import csv
import numpy as np
from itertools import izip
from scipy import stats

def mapping_parse (samp_meta_file, startcol=17, endcol=100, delimiter='\t'):
    """
    INPUTS
    samp_meta_file: file object pointing to a table relating samples 
                    to metabolite levels
    startcol:       integer corresponding to the first column containing 
                    metabolite data in mapping file
    endcol:         integer corresponding to the column AFTER the last 
                    column containing metabolite data in mapping file
    delimiter:      string character that delimites file
    
    OUTPUTS
    samp_ids:   list of strings of sample IDs
    meta_names: list of strings of metabolite names 
    samp_meta:  dict where key = samp id (string) and entry is a list of 
                floats containing metabolite levels
    
    FUNCTION
    Reads in a sample and metabolite data file and returns a list of sample ids 
    (samp_ids), a list of metabolites (meta_names), and a dict mapping 
    samp_ids to metabolite levels.
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
    delimiter:      string character that delimites file
    skip:           number of lines to skip in parsing the otu file (e.g. to 
                    bypass metadata/info in headers) 
    
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
    INPUTS
    ftype: specific string (map or otu) that determines which parsing 
           functionality to perform on the given file
    fp:    string file path
    startcol:       integer corresponding to the first column containing 
                    metabolite data in mapping file
    endcol:         integer corresponding to the column AFTER the last 
                    column containing metabolite data in mapping file
    delimiter:      string character that delimites file
    skip:           number of lines to skip in parsing the otu file (e.g. to 
                    bypass metadata/info in headers) 

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
    """
    
    # initialize matrix; # rows = # of samp_ids, # cols = # entries per key
    rows = len(samp_ids)
    cols = len(samp_dict[samp_dict.keys()[0]])
    samp_matrix = np.zeros(shape=(rows,cols))    
    
    # populate matrix from the dict
    for r in xrange(rows):
        for c in xrange(cols):
            samp_matrix[r][c] = samp_dict[samp_ids[r]][c]

    # retrieve mean values and normalize
    avg_matrix = np.array([np.nanmean(samp_matrix,0)])
    norm_avg_matrix = avg_matrix - avg_matrix.min()
    norm_avg_matrix = norm_avg_matrix/norm_avg_matrix.max()

    # retrieve variances and normalize
    var_matrix = np.array([np.nanvar(samp_matrix,0)])
    norm_var_matrix = var_matrix - var_matrix.min()
    norm_var_matrix = norm_var_matrix/var_matrix.max()

    skew_matrix = np.array([[stats.skew(samp_matrix[:,x],nan_policy='omit') \
        for x in xrange(cols)]])
    return samp_matrix, avg_matrix, norm_avg_matrix,var_matrix,norm_var_matrix, skew_matrix
