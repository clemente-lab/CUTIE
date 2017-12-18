#!/usr/bin/env python
from __future__ import division
    
import os
import numpy as np

def print_matrix(matrix, output_fp, delimiter = '\t', header = []):
    """
    INPUTS
    matrix:     np 2D array to print
    output_fp:  output file path, string
    header:     list of strings to use as header, must match length of matrix

    OUTPUTS
    None

    FUNCTION
    Takes a 2D matrix and writes it to a file
    """
    # check of file exists and if so, remove it before writing
    if os.path.exists(output_fp):
        os.remove(output_fp)

    # obtain dimensions
    rows = np.size(matrix, 0)
    cols = np.size(matrix, 1)

    with open(output_fp, 'w') as f:
        for h in header:
            f.write(h + delimiter)
        f.write('\n')
        for r in xrange(rows):
            for c in xrange(cols):
                f.write(str(matrix[r][c]) + delimiter)
            f.write('\n')


def print_Rmatrix(avg_var1, norm_avg_var1, avg_var2, norm_avg_var2, var_var1,
    norm_var_var1, var_var2, norm_var_var2, skew_var1, skew_var2, n_var1, n_var2,
    variable_names, variables, working_dir, resample_k, label, n_corr, 
    statistic = 'kpc', paired = False):
    """
    Prints results in matrix for R graphing
    *Still need to implement resample_k functionality; currently only works for 
    k = 1
    """
    # create header row
    headers = ['var1_index','var2_index','avg_var1','norm_avg_var1','avg_var2',
                'norm_avg_var2', 'var_var1','norm_var_var1','var_var2',
                'norm_var_var2','skew_var1','skew_var2']

    for var in variable_names:
        headers.append(var)

    # create matrix locally in python
    R_matrix = np.zeros([n_corr, len(headers)])
    row = 0
    for var1 in xrange(n_var1):
        for var2 in xrange(n_var2):
            if not (paired and (var1 == var2)):
                entries = [var1, var2, avg_var1[0][var1], norm_avg_var1[0][var1], 
                            avg_var2[0][var2], norm_avg_var2[0][var2],
                            var_var1[0][var1], norm_var_var1[0][var1],
                            var_var2[0][var2], norm_var_var2[0][var2],
                            skew_var1[0][var1],skew_var2[0][var2]]
                for var in variables:
                    entries.append(var[var1][var2])
                R_matrix[row] = np.array([entries])
                row += 1

    print_matrix(R_matrix, working_dir + 'data_processing/R_matrix_' + label + \
                        '_resample_' + str(resample_k) + '.txt', '\t', headers)


def create_json_matrix(n_var1, n_var2, n_corr, headers, paired, infln_metrics, 
    infln_dict, initial_sig, TP):
    """
    Creates json_matrix for drawing UpSetR plots
    TP is an indicator that indicates whether you are doing TP (1) or FP (0)
    """
    json_matrix = np.zeros([n_corr, len(headers)])

    row = 0
    for var1 in xrange(n_var1):
        # the condition ensures that if calculating auto-correlations (paired == True)
        # then the matrix will not contain entries where var1 == var2
        for var2 in xrange(n_var2):
            if not (paired and (var1 == var2)):
                point = (var1, var2)
                line = [row]
                for metric in infln_metrics:
                    if point in initial_sig:
                        if sum(infln_dict[metric], 0)[var1][var2] == 0:
                            line.append(TP)
                        else:
                            line.append(1 - TP)
                    else:
                        line.append(0)
                TP_json_matrix[row] = np.array([line])
                row += 1

    return json_matrix


def print_json_matrix(n_var1, n_var2, n_corr, infln_metrics, infln_mapping,
    infln_dict, initial_sig, working_dir, paired = False):
    """
    Prints json_matrix for drawing UpSetR plots
    """
    # create header row
    headers = ['corr_row']
    for metric in infln_metrics:
        headers.append(metric)

    # create TP matrix
    TP_json_matrix = create_json_matrix(n_var1, n_var2, n_corr, headers, paired, 
        infln_metrics, infln_dict, initial_sig, 1)
    print_matrix(TP_json_matrix, 
        working_dir + 'data_processing/TP_json_matrix.txt', ';', headers)
    
    # create FP matrix 
    FP_json_matrix = create_json_matrix(n_var1, n_var2, n_corr, headers, paired, 
        infln_metrics, infln_dict, initial_sig, 0)
    print_matrix(FP_json_matrix, 
        working_dir + 'data_processing/FP_json_matrix.txt', ';', headers)
    

def print_prop(n_bact, working_dir, prop, samp_bact_mr, samp_bact_clr, 
    samp_bact_varlog, samp_bact_lclr):
    """
    Prints statistics associated with proportionality variable
    """
    header = [str(x+1) for x in xrange(n_bact)]
    print_matrix(prop, working_dir + 'data_processing/prop.txt', header)
    print_matrix(samp_bact_mr, working_dir + 'data_processing/samp_bact_mr.txt', header)
    print_matrix(samp_bact_clr, working_dir + 'data_processing/samp_bact_clr.txt', header)
    print_matrix(np.array([samp_bact_varlog]), working_dir + 'data_processing/samp_bact_varlog.txt', header)
    print_matrix(samp_bact_lclr, working_dir + 'data_processing/samp_bact_lclr.txt', header)
