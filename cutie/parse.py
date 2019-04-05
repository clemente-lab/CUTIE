#!/usr/bin/env python
from __future__ import division

import matplotlib
matplotlib.use('Agg')

import re
import sys
import os
import csv
import configparser
import numpy as np
import pandas as pd
import hashlib
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

def parse_input(ftype, fp, startcol, endcol, delimiter, skip, log_fp):
    """
    Parses data in OTU-table format (samples as cols, taxa/variables as rows) or
    data in traditional 'mapping' format (samples as rows, variables as
    cols). Span of columns must be continuous. Variables must be
    numeric. Returns dataframe.
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
    samp_ids      - List of strings. Contains sample names in order that they
                    were read.
    var_names     - List of strings. Contains variable names in order that they
                    were read.
    df            - Dataframe. Long/tidy format.
    n_var         - Integer. Number of variables read in.
    n_samp        - Integer. Number of samples read in.
    """

    # read in df and set index
    df = pd.read_csv(fp, sep=delimiter, skiprows=skip, engine='python')
    df = df.set_index(list(df)[0])

    # remove completely NA rows or vars
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)

    # otu tables require transposition
    if ftype == 'otu':
        df = df.T

    # -1 are the defaults; if no start and endcols are specified, read in all cols
    if startcol != -1 or endcol != -1:
        df = df.iloc[:, (startcol-1):(endcol-1)]

    # obtain list of sample ids, variable names, number of var, and number of samples
    samp_ids = df.index.values
    var_names = list(df)
    n_var = len(list(df))
    n_samp = len(df)

    # write to log file
    with open(log_fp, 'a') as f:
        f.write('\nThe length of variables is ' + str(n_var))
        f.write('\nThe number of samples is ' + str(n_samp))

    return samp_ids, var_names, df, n_var, n_samp


def process_df(samp_var_df, samp_ids):
    """
    Reads in dataframe. Returns matrix of values as well as
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
    var_matrix      - 1D array, k-th entry is unbiased variance for variable k.
    """
    # subset dataframes
    samp_var_df = samp_var_df.loc[samp_ids]

    # coerce to NA
    samp_var_df = samp_var_df.apply(pd.to_numeric, errors='coerce')

    # obtain values
    samp_var = samp_var_df.values

    # obtain average and variance
    avg_var = np.array([np.nanmean(samp_var, 0)])

    # retrieve variances
    var_var = np.array([np.nanvar(samp_var, 0)])

    return samp_var, avg_var, var_var


def read_taxa(taxa, delim=';'):
    """
    Converts string of OTU names (e.g. from QIIME) to shortened form.
    ----------------------------------------------------------------------------
    INPUTS
    taxa  - String. Long name of OTU e.g. 'k__Archaea;p__Crenarchaeota;
            c__Thaumarchaeota;o__Cenarchaeales;f__Cenarchaeaceae;
            g__Nitrosopumilus' which will become 'Cenarchaeaceae Nitrosopumilus'
    delim - String. Separates heirarchy.

    OUTPUTS
    String. Abridged taxa name.
    """
    parts = taxa.split(delim) # set as param with default
    if len(parts) > 1:
        while parts:
            if not parts[-1].endswith('__'):
                t1 = parts[-2].split('__')[1]
                t2 = parts[-1].split('__')[1]
                return t1 + ' ' + t2
            else:
                parts.pop()
    else:
        return parts[0]

    # This should not be reached: "k__;p__..."
    return 'Uncharacterized'

###
# MINE parsing
###

def parse_minep(pvalue_fp, delimiter=',', pskip=13):
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
    for i in range(pskip):
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
    # Config = ConfigParser.ConfigParser()
    Config = configparser.ConfigParser()
    Config.read(defaults_fp)
    Config.read(config_fp)

    # [input]
    samp_var1_fp = Config.get('input', 'samp_var1_fp')
    # https://mentaljetsam.wordpress.com/2007/04/13/unescape-a-python-escaped-string/
    delimiter1 = Config.get('input', 'delimiter1')#.decode('string_escape')
    #delimiter1 = Config.get('input', 'delimiter1').decode('string_escape')
    samp_var2_fp = Config.get('input', 'samp_var2_fp')
    delimiter2 = Config.get('input', 'delimiter2')#.decode('string_escape')
    #delimiter2 = Config.get('input', 'delimiter2').decode('string_escape')
    f1type = Config.get('input', 'f1type')
    f2type = Config.get('input', 'f2type')
    skip1 = Config.getint('input', 'skip1')
    skip2 = Config.getint('input', 'skip2')
    minep_fp = Config.get('input', 'minep_fp')
    pskip = Config.getint('input', 'pskip')
    mine_delimiter = Config.get('input', 'mine_delimiter')#.decode('string_escape')
    #mine_delimiter = Config.get('input', 'mine_delimiter').decode('string_escape')
    startcol1 = Config.getint('input', 'startcol1')
    endcol1 = Config.getint('input', 'endcol1')
    startcol2 = Config.getint('input', 'startcol2')
    endcol2 = Config.getint('input', 'endcol2')
    paired = Config.getboolean('input', 'paired')
    overwrite = Config.getboolean('input', 'overwrite')

    # [output]
    label = Config.get('output', 'label')
    working_dir = Config.get('output', 'working_dir')
    log_dir = Config.get('output', 'log_dir')

    # [stats]
    statistic = Config.get('stats', 'statistic')
    resample_k = Config.getint('stats', 'resample_k')
    alpha = Config.getfloat('stats', 'alpha')
    mc = Config.get('stats', 'mc')
    fold = Config.getboolean('stats', 'fold')
    fold_value = Config.getfloat('stats', 'fold_value')
    n_replicates = Config.getint('stats', 'n_replicates')
    log_transform1 = Config.getboolean('stats', 'log_transform1')
    log_transform2 = Config.getboolean('stats', 'log_transform2')
    CI_method = Config.get('stats', 'ci_method')
    sim = Config.getboolean('stats', 'sim')
    corr_path = Config.get('stats', 'corr_path')
    corr_compare = Config.getboolean('stats', 'corr_compare')

    # [graph]
    graph_bound = Config.getint('graph', 'graph_bound')
    fix_axis = Config.getboolean('graph', 'fix_axis')

    return (label, samp_var1_fp, delimiter1, samp_var2_fp, delimiter2, f1type,
            f2type, minep_fp, pskip, mine_delimiter, working_dir, skip1,
            skip2, startcol1, endcol1, startcol2, endcol2, statistic,
            corr_compare, resample_k, paired, overwrite, alpha, mc, fold,
            fold_value, n_replicates, log_transform1, log_transform2, CI_method,
            sim, corr_path, graph_bound, log_dir, fix_axis)
