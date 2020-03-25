#!/usr/bin/env python
import configparser
import hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')

def parse_input(ftype, fp, startcol, endcol, delimiter, skip):
    """
    Parses data in traditional tidy format (samples as rows, variables as
    cols) or untidy/OTU-table format (samples as cols, taxa/variables as rows)
    Span of columns must be continuous. Variables must be numeric.
    Returns dataframe.
    ----------------------------------------------------------------------------
    INPUTS
    ftype       - String. Must be 'tidy' or 'untidy' which specifies parsing
                  functionality to perform on the given file
    fp          - File object. Points to data file.
    startcol    - Integer. Corresponds to first column to include containing
                  variable data in dataframe file.
    endcol      - Integer. Corresponds to the column AFTER the last column
                  containing variable data in dataframe file. Startcol and endcol
                  only relevant if data is in tidy format.
    skip        - Integer. Number lines to skip in parsing the file
    delimiter   - String. Character that delimites file.

    OUTPUTS
    samp_ids      - List of strings. Contains sample names in order that they
                    were read.
    var_names     - List of strings. Contains variable names in order that they
                    were read.
    df            - Dataframe. Wide/tidy format.
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
    if ftype == 'untidy':
        df = df.T

    # -1 are the defaults; if no start and endcols are specified, read in all
    # cols, else subset
    if startcol != -1 and endcol != -1:
        df = df.iloc[:, (startcol-1):(endcol-1)]
    elif not (startcol == -1 and endcol == -1):
        raise ValueError('Both startcol and endcol must be specified')
    # obtain list of sample ids, variable names, number of var, and number of
    # samples
    samp_ids = df.index.values
    var_names = [str(x) for x in list(df)]
    n_var = len(list(df))
    n_samp = len(df)

    return samp_ids, var_names, df, n_var, n_samp

def process_df(samp_var_df, samp_ids):
    """
    Reads in dataframe. Returns matrix of values. Nans are ignored in all cases.
    ----------------------------------------------------------------------------
    INPUTS
    samp_var_df     - Dataframe. Index is sample id and columns are variables
                      of metadata.
    samp_ids        - List of strings. Contains sample names in order that they
                      were read.

    OUTPUTS
    samp_var        - 2D array where each value in row i col j is the level of
                      variable j corresponding to sample i in the order that
                      the samples are presented in samp_ids
    """
    # subset dataframes
    samp_var_df = samp_var_df.loc[samp_ids]

    # coerce NA's
    samp_var_df = samp_var_df.apply(pd.to_numeric, errors='coerce')

    # obtain values
    samp_var = samp_var_df.values

    return samp_var

###
# Config parsing
###
def parse_config(input_config_fp):
    """
    Config parser to unpack arguments for CUtIe.
    ----------------------------------------------------------------------------
    INPUTS
    config_fp   - File object. Contains specific config settings for a given
                  execution.

    OUTPUTS
    List of variables corresponding to arguments for calculate_cutie.py.
    """
    defaults = {
        'delimiter1': '\t',
        'delimiter2': '\t',
        'f1type': 'tidy',
        'f2type': 'tidy',
        'skip1': 0,
        'skip2': 0,
        'startcol1': -1,
        'endcol1': -1,
        'startcol2': -1,
        'endcol2': -1,
        'paired': False,
        'overwrite': True,
        'statistic': 'pearson',
        'k': 1,
        'alpha': 0.05,
        'mc': 'fdr',
        'fold': False,
        'fold_value': 1,
        'corr_compare': False,
        'graph_bound': 30,
        'fix_axis': False
    }
    Config = configparser.ConfigParser(defaults=defaults)
    Config.read(input_config_fp)

    # [input]
    samp_var1_fp = Config.get('input', 'samp_var1_fp')
    delimiter1 = Config.get('input', 'delimiter1')
    samp_var2_fp = Config.get('input', 'samp_var2_fp')
    delimiter2 = Config.get('input', 'delimiter2')
    f1type = Config.get('input', 'f1type')
    f2type = Config.get('input', 'f2type')
    skip1 = Config.getint('input', 'skip1')
    skip2 = Config.getint('input', 'skip2')
    startcol1 = Config.getint('input', 'startcol1')
    endcol1 = Config.getint('input', 'endcol1')
    startcol2 = Config.getint('input', 'startcol2')
    endcol2 = Config.getint('input', 'endcol2')
    paired = Config.getboolean('input', 'paired')
    overwrite = Config.getboolean('input', 'overwrite')

    # [output]
    working_dir = Config.get('output', 'working_dir')

    # [stats]
    param = Config.get('stats', 'param')
    statistic = Config.get('stats', 'statistic')
    resample_k = Config.getint('stats', 'resample_k')
    alpha = Config.getfloat('stats', 'alpha')
    multi_corr = Config.get('stats', 'mc')
    fold = Config.getboolean('stats', 'fold')
    fold_value = Config.getfloat('stats', 'fold_value')
    corr_compare = Config.getboolean('stats', 'corr_compare')

    # [graph]
    graph_bound = Config.getint('graph', 'graph_bound')
    fix_axis = Config.getboolean('graph', 'fix_axis')

    return (samp_var1_fp, delimiter1, samp_var2_fp, delimiter2, f1type,
            f2type, working_dir, skip1, skip2, startcol1, endcol1, startcol2,
            endcol2, param, statistic, corr_compare, resample_k, paired,
            overwrite, alpha, multi_corr, fold, fold_value, graph_bound, fix_axis)

def md5_checksum(fp):
    """
    Computes the md5 of a given file (for log purposes).
    https://www.joelverhagen.com/blog/2011/02/md5-hash-of-file-in-python/
    ----------------------------------------------------------------------------
    INPUTS
    fp - String. Directory and name of file to be evaluated.

    OUTPUTS
    Returns string of file md5
    """
    with open(fp, 'rb') as f:
        m = hashlib.md5()
        while True:
            data = f.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()