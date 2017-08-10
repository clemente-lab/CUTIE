#!/usr/bin/env python
from __future__ import division

import re
import sys
import os
import numpy as np

def samp_meta_parse (samp_meta_file, 
                     startcol=17, 
                     endcol=100):
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
    
    line_number = 0
    for line in samp_meta_file:
        # generate metabolite list from the 0th line (header)
        # default assumes metabolites are in col 17 to 99
        if line_number is 0:
            meta_names = line.split('\t')[startcol:endcol]
            line_number += 1
        else: 
            # for the remainder of the lines (i.e. the non-header lines)
            line = line.split('\t')
            samp_ids.append(line[0]) # line[0] is the sample id
            metabolite_levels = line[startcol:endcol]
            samp_meta[line[0]] = metabolite_levels
   
    return samp_ids, meta_names, samp_meta

def samp_bact_parse(samp_bact_file):
    """ 
    INPUTS
    samp_bact_file: file object pointing to an OTU table of bacteria levels
    
    OUTPUTS
    bact_names: list of strings corresponding to bacteria names 
                (in the OTU table)
    samp_bact:  dict where key = sample id (string) and entry is a string
                containing relative abundance of bacteria
    
    FUNCTION
    Reads in a sample-bacteria file (OTU table) and returns a list of bacteria 
    names (bact_names) and a dict (samp_bact) mapping sample ids to relative
    abundances.
    
    EXAMPLE
    with open(
            'data/otu_table_MultiO__Status_merged___L6.txt',
            'r') 
        as f,
        bact_list_L6 = subj_bact_merge_parse(f)
    """
    
    # create lists (corresponding to smoking and non-smoking files) 
    bact_names = []
    samp_bact = {}
    
    samp_bact_file.readline() # line 0 is 'constructed from biom file' 
    samp_ids = samp_bact_file.readline().rstrip().split('\t')
    samp_ids.pop(0) # the 0th entry is a header
    for samp_id in samp_ids:
        samp_bact[samp_id] = []

    for line in samp_bact_file:
        if line is not '': 
            split_line = line.rstrip().split('\t')
            # the 0th entry is the name of an OTU
            bact_names.append(split_line[0])
            split_line.pop(0) # pop off OTU
            for b in xrange(len(split_line)):
                samp_bact[samp_ids[b]].append(split_line[b])
        
    return bact_names, samp_bact

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
    for r in xrange(0,rows):
        for c in xrange(0,cols):
            samp_matrix[r][c] = samp_dict[samp_ids[r]][c]
    return samp_matrix
