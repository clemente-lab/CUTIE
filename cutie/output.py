#!/usr/bin/env python
from __future__ import division
    
import os
import numpy as np
from scipy import stats
import statsmodels.api as sm

def print_stats_SLR(
        stat_dict,
        working_dir,
        label,
        functions,
        mapf,
        f_stats
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
    mapf = {
        'stats.linregress': stats.linregress,
        'stats.spearmanr': stats.spearmanr
        }
    f_stats = {'stats.linregress': 
                   ['b1', 'b0', 'pcorr','ppvalue','stderr'],
               'stats.spearmanr':
                   ['scorr','spvalue']}
    print_stats_SLR(stat_dict,
                    working_dir,
                    label = 'L6',
                    functions,
                    mapf,
                    f_stats)
    """
    # functions: list of function objects to be called
    # function_stats: dict where key is function and elements are strings
    #     of relevant statistics corresponding to output of function
    
    stat_files = {}
    
    num_bact = np.size(stat_dict[stat_dict.keys()[0]], 1)
    num_meta = np.size(stat_dict[stat_dict.keys()[0]], 2)
    
    # retrieve relevant stats
    for f in functions: # e.g. f = 'stats.linregress'
        rel_stats = f_stats[f] # e.g. ['b1', 'b0', 'pcorr','ppvalue','stderr']
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
                    stat_files[f][rel_stats[r]].write(str(matrix[r][b][m]) + '\t')
            # print a new line after every bacteria
            for r in xrange(num_rel):
                stat_files[f][rel_stats[r]].write('\n')
        # close the files after all bacteria are done
        for r in xrange(num_rel):
            stat_files[f][rel_stats[r]].close()
    return


def print_matrix(matrix, output_fp, header = []):
    """
    INPUTS
    matrix:     np 2D array to print
    output_fp:  output file path, string
    header:     list of strings to use as header, must match length of matrix

    OUTPUTS
    None

    FUNCTION
    Takes a 2D matrix and writes it to a file

    EXAMPLE
    output.print_matrix(avg_meta_matrix, 
                            working_dir + 'data_processing/meta_avg_' + label + '.txt')
    """
    if os.path.exists(output_fp):
        os.remove(output_fp)

    rows = np.size(matrix, 0)
    cols = np.size(matrix, 1) 
    with open(output_fp, 'w') as f:
        for h in header:
            f.write(h + '\t')
        f.write('\n')
        for r in xrange(rows):
            for c in xrange(cols):
                f.write(str(matrix[r][c]) + '\t')
            f.write('\n')

    return