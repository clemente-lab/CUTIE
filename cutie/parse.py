#!/usr/bin/env python
from __future__ import division
 import re
import sys
import os
import csv
import ConfigParser
import numpy as np
import hashlib
from itertools import izip
from scipy import stats

def md5Checksum(filePath):
    """
    Computes the md5 of a given file (for log purposes).
    https://www.joelverhagen.com/blog/2011/02/md5-hash-of-file-in-python/
    ----------------------------------------------------------------------------
    INPUTS
    filePath - String. Directory and name of file to be evaluated.
     OUTPUTS
    Returns string of file md5
    """
    with open(filePath, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()
 def mapping_parse (samp_var_file, log_fp, startcol, endcol, skip, delim = '\t'):
    """
    Parses data in traditional 'mapping' format (samples as rows, variables as 
    cols), returning a dict where key = sample name and entry = array of 
    variable levels. Span of columns must be continuous. Variables must be 
    numeric.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var_file - File object. Points to data relating samples to variables.
    log_fp        - File object. Points to log file. 
    startcol      - Integer. Corresponds to first column to include containing 
                    variable data in mapping file.
    endcol        - Integer. Corresponds to the column AFTER the last 
                    column containing variable data in mapping file. 
    skip          - Integer. Number lines to skip in parsing the file 
    delim         - String. Character that delimites file.
    
    OUTPUTS
    samp_ids      - List of strings. Contains sample names in order that they 
                    were read.
    var_names     - List of strings. Contains variable names in order that they
                    were read.
    samp_to_var   - Dictionary. Key = sample id (from samp_ids) and entry is a 
                    list of floats containing variable levels
    n_var         - Integer. Number of variables read in.
    n_samp        - Integer. Number of samples read in.
    """
    samp_ids = []
    samp_to_var = {}
    
    # skip rows
    for i in xrange(skip):
        samp_var_file.readline()
     # generate variable list from the 0th line (header)
    # example header:
    # #SampleID    glutamic_acid    glycine
    # example line:
    # 100716FG.C.1.RL    60088.1214    811001.3592
     # process header
    var_names = samp_var_file.readline().split(delim)[startcol:endcol]
    # process reamining non-header lines
    for line in samp_var_file:
        # check if line is empty
        if line != '\n':
            line = line.split('\n')[0].split(delim)
            # line[0] is the sample id
            samp_ids.append(line[0]) 
            # empty values are replaced with np.nan
            if startcol == -1 and endcol == -1:
                var_values = [np.nan if x == '' else float(x) for x in line[1:]]
            else:
                var_values = [np.nan if x == '' else float(x) \
                    for x in line[startcol:endcol]]
            # if there is trailing empty cells, fill with np.nan
            while len(var_values) < len(var_names):
                var_values.append(np.nan)
            # populate dict
            samp_to_var[line[0]] = var_values
    n_var = len(var_names)
    n_samp = len(samp_ids)
     # write to log file
    with open(log_fp, 'a') as f:
        f.write('\nThe length of mapping_variables is ' + str(n_var))
        f.write('\nThe number of samples is ' + str(n_samp))
     return samp_ids, var_names, samp_to_var, n_var, n_samp
 def otu_parse(samp_var_file, log_fp, skip, delim = '\t'):
    """
    Parses data in OTU-table format (samples as cols, taxa/variables as rows),
    returning a dict where key = sample name and entry = array of taxa/variable 
    levels. Variables must be numeric.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var_file - File object. Points to data relating samples to variables.
    log_fp        - File object. Points to log file. 
    skip          - Integer. Number lines to skip in parsing the file 
    delimiter     - String. Character that delimites file.
    
    OUTPUTS
    samp_ids      - List of strings. Contains sample names in order that they 
                    were read.
    var_names     - List of strings. Contains variable names in order that they
                    were read.
    samp_to_var   - Dictionary. Key = sample id (from samp_ids) and entry is a 
                    list of floats containing variable/taxa levels
    n_var         - Integer. Number of variables read in.
    n_samp        - Integer. Number of samples read in.
    """    
    var_names = []
    samp_to_var = {}
    
    # skip rows
    # In QIIME summarized OTU tables, line 0 is '# Constructed from biom file' 
    for i in xrange(skip):
        samp_var_file.readline() 
     # generate sample list from the 0th line (header)
    # example header:
    # #OTU ID    101019AB.N.1.RL    110228CJ.N.1.RL    110314CS.N.1.RL 
    # for the remainder of the lines (i.e. the non-header lines)
    # example line:
    # k__Archaea;...;g__    0   0.008736039    0   
    
    # process header
    samp_ids = samp_var_file.readline().rstrip().split(delim)
    samp_ids.pop(0) # the 0th entry is a #OTU ID
    for samp_id in samp_ids:
        samp_to_var[samp_id] = []
     # process remaining lines
    for line in samp_var_file:
        if line is not '': 
            split_line = line.rstrip().split(delim)
            # the 0th entry is the name of an OTU
            var_names.append(split_line[0])
            split_line.pop(0) # pop off OTU
            for b in xrange(len(split_line)):
                samp_to_var[samp_ids[b]].append(split_line[b])
        
    n_var = len(var_names)
    n_samp = len(samp_ids)
     # write to log file
    with open(log_fp, 'a') as f:
        f.write('\nThe length of mapping_variables is ' + str(n_var))
        f.write('\nThe number of samples is ' + str(n_samp))
            
    return samp_ids, var_names, samp_to_var, n_var, n_samp
 def parse_input(ftype, fp, startcol, endcol, delimiter, skip, log_fp):
    """
    Parses data in OTU-table format (samples as cols, taxa/variables as rows),
    returning a dict where key = sample name and entry = array of taxa/variable 
    levels. Variables must be numeric.
    ----------------------------------------------------------------------------
    INPUTS
    ftype       - String. Must be 'map' or 'otu' which specifies parsing 
                  functionality to perform on the given file
    fp          - File object. Poiints to data file.
    startcol    - Integer. Corresponds to first column to include containing 
                  variable data in mapping file.
    endcol      - Integer. Corresponds to the column AFTER the last column 
                  containing variable data in mapping file. Startcol and endcol 
                  only relevant if data is in mapping format.
    skip        - Integer. Number lines to skip in parsing the file 
    delimiter   - String. Character that delimites file.
    log_fp      - File object. Points to log file. 
    
    OUTPUTS
    samp_ids    - List of strings. Contains sample names in order that they 
                    were read.
    var_names   - List of strings. Contains variable names in order that they
                    were read.
    samp_to_var - Dictionary. Key = sample id (from samp_ids) and entry is a 
                    list of floats containing variable/taxa levels
    n_var       - Integer. Number of variables read in.
    n_samp      - Integer. Number of samples read in.
    """    
    # fork depending on ftype
    # some files won't split on \n but will on \rU
    if ftype == 'map':
        with open(fp,'rU') as f:    
            samp_ids, var_names, samp_to_var, n_var, n_samp = \
                mapping_parse(f, log_fp, startcol, endcol, skip, delimiter)
     elif ftype == 'otu':
        with open(fp, 'rU') as f:
            samp_ids, var_names, samp_to_var, n_var, n_samp = \
                otu_parse(f, log_fp, skip, delimiter)     
     return samp_ids, var_names, samp_to_var, n_var, n_samp 
 def dict_to_matrix(samp_to_var, samp_ids):
    """
    Reads in dict where the key is the sample id and the subsequent values are 
    variable levels and loads it into an np.array where row i and column j 
    correspond to the value of sample i, the rows being sorted in the order that 
    they are inputted per samp_ids. Returns matrix of results as well as
    matrices for average and normalized mean and variance of each variable, 
    as well as (unnormalized) skew. Nans are ignored in all cases.
    ----------------------------------------------------------------------------
    INPUTS
    samp_to_var     - Dictionary. Key = sample id (from samp_ids) and entry is a 
                      list of floats containing variable/taxa levels
    samp_ids        - List of strings. Contains sample names in order that they 
                      were read.
    
    OUTPUTS
    samp_var        - 2D array where each value in row i col j is the level of 
                      variable j corresponding to sample i in the order that
                      the samples are presented in samp_ids
    avg_matrix      - 1D array where k-th entry is mean value for variable k. 
                      Variables are ordered as in original data file (i.e. order 
                      is presered through parsing). 
    norm_avg_matrix - Normalized (values between 0 to 1) avg_matrix. 
    var_matrix      - 1D array where k-th entry is unbiased variance for variable k. 
    norm_var_matrix - Normalized (values between 0 to 1) var_matrix.
    skew_matrix     - 1D array where k-th entry is skew for variable k. 
    """    
    
    # initialize matrix; # rows = # of samp_ids, # cols = # entries per key
    rows = len(samp_ids)
    cols = len(samp_to_var[samp_to_var.keys()[0]])
    samp_var = np.zeros(shape=(rows,cols))    
    
    # populate matrix from the dict
    for r in xrange(rows):
        for c in xrange(cols):
            samp_var[r][c] = samp_to_var[samp_ids[r]][c]
     # retrieve mean values and normalize
    avg_matrix = np.array([np.nanmean(samp_var,0)])
    norm_avg_matrix = avg_matrix - avg_matrix.min()
    norm_avg_matrix = norm_avg_matrix/norm_avg_matrix.max()
     # retrieve variances and normalize
    var_matrix = np.array([np.nanvar(samp_var,0)])
    norm_var_matrix = var_matrix - var_matrix.min()
    norm_var_matrix = norm_var_matrix/var_matrix.max()
     # retrieve skew values
    skew_matrix = np.array([[stats.skew(samp_var[:,x], nan_policy='omit') \
        for x in xrange(cols)]])
     return (samp_var, avg_matrix, norm_avg_matrix, var_matrix, 
        norm_var_matrix, skew_matrix)
 def read_taxa(taxa):
    """
    Converts string of OTU names (e.g. from QIIME) to shortened form.
    ----------------------------------------------------------------------------
    INPUTS
    taxa - String. Long name of OTU e.g. 'k__Archaea;p__Crenarchaeota;
           c__Thaumarchaeota;o__Cenarchaeales;f__Cenarchaeaceae;
           g__Nitrosopumilus' which will become 'Cenarchaeaceae Nitrosopumilus'
    
    OUTPUTS
    String. Abridged taxa name.
    """    
    parts = taxa.split(';')
    while parts:
        if not parts[-1].endswith('__'):
            t1 = parts[-2].split('__')[1]
            t2 = parts[-1].split('__')[1]
            return t1 + ' ' + t2
        else:
            parts.pop()
     # This should not be reached: "k__;p__..."
    return 'Uncharacterized'
 ###
# MINE parsing
###
 def parse_minep(pvalue_fp, delimiter = ',', pskip = 13):
    """
    Parses MINE downloaded pvalue file e.g. from http://www.exploredata.net/ 
    that contains table of pvalue-MIC_strength relationship provided by MINE 
    developers. Choose file depending on sample size!
    ----------------------------------------------------------------------------
    INPUTS
    pvalue_fp   - File object. Points to pvalue file. 
    delimiter   - String. Default is ',' as MINE uses csv files by default.
    pskip       - Integer. Number of rows to skip in the pvalue table (default is 
                  13, to bypass various comments)
    
    OUTPUTS
    MINE_bins   - 2D Array. Each row is in format [MIC_str, pvalue, stderr of 
                  pvalue]. Pvalue corresponds to probability of observing 
                  MIC_str as or more extreme as observed MIC_str.
    pvalue_bins - List. Sorted list of pvalues from greatest to least used 
                  by MINE to bin the MIC_str.
    """
    MINE_bins = []
    pvalue_bins = []
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
            pvalue_bins.append(row[0]) # row[0] is the pvalue
     # convert list to array
    MINE_bins = np.array(MINE_bins)
     return MINE_bins, pvalue_bins 
 ###
# Config parsing
###
def parse_config(defaults_fp, config_fp):
    """
    Config parser to unpack arguments for CUtIe.
    ----------------------------------------------------------------------------
    INPUTS
    defaults_fp - File object. Contains default settings (config_defaults.ini).
    config_fp   - File object. Contains specific config settings for a given 
                  execution.
     OUTPUTS
    List of variables corresponding to arguments for calculate_cutie.py.
    """    
    Config = ConfigParser.ConfigParser()
    Config.read(defaults_fp)
    Config.read(config_fp)
     # [input]
    samp_var1_fp = Config.get('input', 'samp_var1_fp')
    # https://mentaljetsam.wordpress.com/2007/04/13/unescape-a-python-escaped-string/
    delimiter1  = Config.get('input', 'delimiter1').decode('string_escape') 
    samp_var2_fp  = Config.get('input','samp_var2_fp')
    delimiter2 = Config.get('input','delimiter2').decode('string_escape') 
    f1type = Config.get('input','f1type')
    f2type = Config.get('input','f2type') 
    skip1 = Config.getint('input','skip1')
    skip2 = Config.getint('input','skip2')
    mine_fp = Config.get('input','mine_fp') 
    minep_fp = Config.get('input','minep_fp')
    pskip = Config.getint('input','pskip')
    mine_delimiter = Config.get('input','mine_delimiter').decode('string_escape') 
    startcol1 = Config.getint('input', 'startcol1')  
    endcol1 = Config.getint('input','endcol1')
    startcol2 = Config.getint('input','startcol2')
    endcol2 = Config.getint('input','endcol2')
    paired = Config.getboolean('input','paired')
     # [output]
    label = Config.get('output','label')
    working_dir = Config.get('output','working_dir')
    log_dir = Config.get('output','log_dir')
     # [stats]
    statistic = Config.get('stats','statistic')
    resample_k = Config.getint('stats', 'resample_k')
    alpha = Config.getfloat('stats', 'alpha')
    mc = Config.get('stats','mc')
    fold = Config.getboolean('stats','fold')
    fold_value = Config.getfloat('stats','fold_value')
    n_replicates = Config.getint('stats','n_replicates')
    log_transform1 = Config.getboolean('stats','log_transform1')
    log_transform2 = Config.getboolean('stats','log_transform2')
    CI_method = Config.get('stats','ci_method')
    sim = Config.getboolean('stats','sim')
    corr_path = Config.get('stats','corr_path')
    corr_compare = Config.getboolean('stats','corr_compare')
    
    # [graph]
    graph_bound = Config.getint('graph','graph_bound')
    
    return (label, samp_var1_fp, delimiter1, samp_var2_fp, delimiter2, f1type, 
        f2type, mine_fp, minep_fp, pskip, mine_delimiter, working_dir, skip1, 
        skip2, startcol1, endcol1, startcol2, endcol2, statistic, corr_compare, 
        resample_k, paired, alpha, mc, fold, fold_value, n_replicates, 
        log_transform1, log_transform2, CI_method, sim, corr_path, graph_bound, 
        log_dir)