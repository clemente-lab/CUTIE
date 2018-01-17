#!/usr/bin/env python
from __future__ import division

from cutie import parse
from cutie import statistics
from cutie import output
from cutie import __version__

import time
import click
import os
import numpy as np
from scipy import stats


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)

@click.option('-l', '--label', required=True,
              help='label to save files as')
@click.option('-s1', '--samp_var1_fp', required=True,
              type=click.Path(exists=True),
              help='Input  samp variable 1 file')
@click.option('-dl1', '--delimiter1', default='\t',
                required=False, help='delimiter for first file')
@click.option('-s2', '--samp_var2_fp', required=True,
              type=click.Path(exists=True),
              help='Input  samp variable 2 file')
@click.option('-dl2', '--delimiter2', default='\t',
                required=False, help='delimiter for second file')
@click.option('--otu1', 'f1type', flag_value='otu',
              default=True, required=True, help='')
@click.option('--map1', 'f1type', flag_value='map',
              default=False, required=True, help='')
@click.option('--map2', 'f2type', flag_value='map',
              default=True, required=True, help='')
@click.option('--otu2', 'f2type', flag_value='otu',
              required=False, help='')
@click.option('-mf', '--mine_fp', required=False,
              type=click.Path(exists=True),
              help='Input  mine file')
@click.option('-mpf', '--minep_fp', required=False,
              type=click.Path(exists=True),
              help='Input  mine pvalue file')
@click.option('-pskip', '--pskip', default=13, required=False,
              type=int, help='rows of pvalue table to skip')
@click.option('-dlm', '--mine_delimiter', default=',',
                required=False, help='delimiter for MINE files')
@click.option('-d', '--working_dir', default='', required=False,
              help='Directory to save files')
@click.option('-skip', '--skip', default=1, required=False,
              type=int, help='row of otu table to skip')
@click.option('-sc', '--startcol', default=17, required=False,
              type=int, help='starting metabolite col')
@click.option('-ec', '--endcol', default=100, required=False,
              type=int, help='ending metabolite col')
@click.option('-stat', '--statistic', default='kpc',
              required=True, help='statistic to use')
@click.option('--point', 'point_compare', is_flag=True,
                flag_value=True,
              required=False, help='True if comparing points')
@click.option('-mnp', '--mine_non_par', default=False, required=False,
                type=bool, help='boolean, true if need to calculate the MINE resampling non parallel')
@click.option('--point', 'point_compare', is_flag=True,
                flag_value=True,
              required=False, help='True if comparing points')
@click.option('--corr', 'corr_compare', is_flag=True,
                flag_value=True,
              required=False, help='True if comparing corr')
@click.option('-k', '--resample_k', default=1, required=False,
              type=int, help='number to resample')
@click.option('-rz', '--rm_zero', default=False, required=False,
              help='True if removing 0s')
@click.option('-np', '--n_parallel', default=1, required=False,
              help='Number of parallel processes to create')
@click.option('-p', '--paired', default=False, required=False,
                type=bool, help='boolean, true if correlating variable to self')
@click.option('-a', '--alpha', default=0.05, required=False,
              type=float, help='threshold value')
@click.option('--nomc', 'mc', flag_value='nomc',
              default=True, required=False,
              help='True if no mc correction used')
@click.option('--bc', 'mc', flag_value='bc',
              required=False, help='True if using Bonferroni correction')
@click.option('--fwer', 'mc', flag_value='fwer',
              required=False, help='True if using FWER correction')
@click.option('--fdr', 'mc', flag_value='fdr',
              required=False, help='True if using FDR correction')
@click.option('--fold', 'fold', is_flag=True,
                flag_value=True,
              required=False, help='True if using new adjustment')



def calculate_cutie(label, samp_var1_fp, delimiter1, samp_var2_fp, delimiter2, 
                    f1type, f2type, mine_fp, minep_fp, pskip, mine_delimiter, 
                    working_dir, skip, startcol, endcol, statistic, mine_non_par,
                    point_compare, 
                    corr_compare, resample_k, rm_zero, n_parallel, paired, alpha, mc, fold):
    """ 
    INPUTS
    label:        string to uniquely tag subsequent output files
    samp_var1_fp: file object pointing to variable 1 data file (OTU table)
    samp_var2_fp: file object pointing to variable 2 data file 
    working_dir:  file path to where data is processed
    startcol:     int, first column in mapping file holding data (for var 2 fp)
    endcol:       int, ONE AFTER last column in mapping data holding data
    statistic:    flag, string of statistic to compute (--kpc for pearson vs 
                  --ksc for spearman)
    k:            int, number of resamples (k <= sample size-3)
    alpha:        threshold used for alpha, 0.05 by default
    mc:           flag, string used to determine which type of MC to use
                    --nomc is no MC at all, --bc is Bonferroni correction,
                    --fdr is false discovery rate correction

    FUNCTION
    Computes pairwise correlations between each bact, meta pair and 
    for the significant correlations, recomputes correlation for each pair
    after iteratively excluding n observations, differentiating
    true and false correlations on the basis of whether the correlation remains
    significant when each individual observation is dropped
    """

    start_time = time.clock()

    ### 
    # Parsing and Pre-processing
    ###

    # create subfolder to hold data analysis files
    if os.path.exists(working_dir + 'data_processing') is not True:
        os.makedirs(working_dir + 'data_processing')
        
    # file handling and parsing decisions
    # file 1 is the 'dominant' file type and should always contain the OTU file
    # we let the dominant fil 'override' the sample_id list ordering
    samp_ids, var2_names, samp_to_var2, n_var2, n_samp = \
        parse.parse_input(f2type, samp_var2_fp, startcol, endcol, delimiter2, skip)
    samp_ids, var1_names, samp_to_var1, n_var1, n_samp = \
        parse.parse_input(f1type, samp_var1_fp, startcol, endcol, delimiter1, skip)  

    # temporary printing of samp and var names for reference
    print samp_ids[0:5]
    print var1_names[0:5]

    # convert dictionaries to matrices
    samp_var1, avg_var1, norm_avg_var1, var_var1, norm_var_var1, skew_var1 = \
        parse.dict_to_matrix(samp_to_var1, samp_ids)
    samp_var2, avg_var2, norm_avg_var2, var_var2, norm_var_var2, skew_var2 = \
        parse.dict_to_matrix(samp_to_var2, samp_ids)



    # if parallel is not one, divide the groups
    # compute cutie on each group and output into file
    # combine output files
    # n_parallel
    '''
    if n_parallel > 1:
        # split on the first variable which is usually the OTU table
        module_chunks = np.array_split(n_var1,n_parallel)
        chunks = []
        for module_chunk in module_chunks:
            # take the first and last entry as bounds
            chunks.append(module_chunk[0],module_chunk[-1])
    '''

    ###
    # Initial Statistics and Update Statistics
    ### 

    # if statistic is pearson or spearman correlation
    if statistic == 'kpc' or statistic == 'ksc':
        # statistic-specific initial output
        if statistic == 'kpc':
            pvalues, corrs, logpvals, r2vals = \
                statistics.assign_statistics(n_var1, n_var2, samp_var1, samp_var2, 
                    statistic, rm_zero)

        elif statistic == 'ksc':
            pvalues, logpvals, corrs = statistics.assign_statistics(
                n_var1, n_var2, samp_var1, samp_var2, statistic, rm_zero)

        # determine significance threshold and number of correlations
        threshold, n_corr = statistics.set_threshold(pvalues, alpha, mc, paired)

        # update statistics via CUtIe resampling
        initial_sig, true_sig, comb_to_rev, worst_p, worst_r = \
            statistics.updatek_cutie_SLR(n_var1, n_var2, n_samp, pvalues, 
                samp_ids, samp_var1, samp_var2, threshold, resample_k, 
                corrs, fold, paired, statistic)

    # if computing proportionality
    elif statistic == 'prop':
        # adjust var1 for proportionality
        samp_var1_mr, samp_var1_clr, samp_var1_lclr, samp_var1_varlog, \
            correction, n_zero = statistics.multi_zeros(samp_var1)
        prop = statistics.initial_stats_prop(samp_var1_clr, samp_var1_varlog)
        output.print_prop(n_var1, working_dir, prop, samp_var1_mr, samp_var1_clr, 
            samp_var1_varlog, samp_var1_lclr)
        n_corr = n_var1 * n_var1

        # arbitrarily chosen prop threshold for now (based on paper)
        prop_threshold = alpha * 10
        # empty dictionary as placeholder
        comb_to_rev = {}

        # update statistics via CUtIe resampling
        initial_sig, true_sig, worst_prop, all_points, headers = \
            statistics.updatek_cutie_prop(n_var1, n_samp, prop, samp_ids, 
                                samp_var1_clr, prop_threshold, resample_k)

        output.print_matrix(all_points, working_dir + 'data_processing/' + \
            'all_points_prop_' + 'resample_' + str(resample_k) + '.txt', headers)


    ###
    # CODE REVIEW: MINE
    ###

    elif statistic == 'mine':
        # input from script: mine_fp = otu_table_small.MSQ34_L6.csv, label = 'otu'
        # 1. create transposed data fn and fp strings
        #   e.g. otu_transpose_table_small.MSQ34_L6.csv
        transposed_fn = label + '_transpose' + mine_fp.split(label)[1]
        transposed_fp = working_dir + transposed_fn

        # transpose input csv file (mine_fp), checking if it exists first
        # transposed_fp = data_analysis/MIC_MSQ_test2/otu_transpose_table_small.MSQ34_L6.csv
        # tr '\r' '\n' <a.txt> b.txt
        if os.path.isfile(transposed_fp) == False:
            with open(mine_fp, "rU") as f:
                parse.transpose_csv(f, transposed_fp, skip)
            
        # obtain MINE output for full dataset 
        # results stored in otu_table_small.MSQ34_L6.csv,allpairs,cv=0.1,B=n^0.6,Results.csv
        # parameters are default, will be changed if needed
        original_mine_fp = working_dir + transposed_fn + \
            ",allpairs,cv=0.1,B=n^0.6,Results.csv"
        if os.path.isfile(original_mine_fp) == False:
            # .. replace with path default
            os.system("java -jar ../MINE/MINE.jar " + transposed_fp + \
                " '-allPairs' cv=0.1 exp=0.6 c=10 fewBoxes")
        

        if mine_non_par == True:
            # subset data files  
            # PP package, forkfun
            parse.subset_data(n_samp, transposed_fn, transposed_fp, working_dir)

            # run MINE on each subset
            statistics.mine_subsets(n_samp, transposed_fn, transposed_fp, \
                                    original_mine_fp, working_dir, label)

        # declare initial stats to parsex
        # provide statistic headers as headers in MINE output are less wieldy
        mine_stats = ['MIC_str','MIC_nonlin','MAS_nonmono','MEV_func',
                      'MCN_comp','linear_corr'] 
        with open(original_mine_fp, 'rU') as f:
            stat_to_matrix = parse.parse_mine(f, n_var1, var1_names, 
                                                mine_stats, mine_delimiter)
    
        with open(minep_fp, 'rU') as f:
            mine_bins, pvalues_ordered = parse.parse_minep(f, mine_delimiter,
                                                             pskip)

        #print mine_bins
        #print pvalues_ordered
        # Store MINE_str and compute pvalue for each correlation 
        mine_str = stat_to_matrix['MIC_str']
        mine_nonlin = stat_to_matrix['MIC_nonlin']
        mine_pvalues = statistics.str_to_pvalues(n_var1, pvalues_ordered, 
                                                    mine_str, mine_bins)

        # Exhaustively parse all subsetted files and store results in 3d arrays
        mine_subset_str, mine_subset_p, mine_subset_nonlin = statistics.subset_mine(
                                n_samp, n_var1, var1_names, label, mine_stats, 
                                pvalues_ordered, mine_bins, working_dir, 
                                original_mine_fp, mine_delimiter)

        # set FDR threshold
        threshold, n_corr = statistics.set_threshold(mine_pvalues, alpha, mc, 
                                                        paired)

        initial_sig, true_sig = statistics.cutie_mine(n_samp, n_var1, threshold, 
                                                    mine_pvalues, mine_subset_p)
        ### initial statistics       
        print 'Length of initial sig MINE is ' + str(len(initial_sig))
        print 'Length of true sig MINE is ' + str(len(true_sig['1']))


        # obtain worst_mine_p value and worst_mine_str for each 
        worst_mine_p = np.amax(mine_subset_p, axis=0)
        worst_mine_str = np.amin(mine_subset_str, axis=0)
        max_mine_subset_nonlin = np.amax(mine_subset_nonlin, axis=0)
        min_mine_subset_nonlin = np.amin(mine_subset_nonlin, axis=0)

        # empty dictionary as placeholder
        comb_to_rev = {}

    else:
        print 'Invalid statistic: ' + statistic + ' chosen'

    ###
    # Determine indicator matrix of significance
    ###

    # element i,j is -1 if flagged by CUtIe, 1 if not, and 0 if insig originally
    indicators = statistics.return_indicators(n_var1, n_var2, initial_sig, 
                                        true_sig, resample_k)

    ###
    # Report statistics
    ###
    for k in xrange(resample_k):
        if statistic == 'kpc' or statistic == 'ksc':
            p_ratio = np.divide(worst_p[str(k+1)], pvalues)

            if statistic == 'kpc':
                r2_ratio = np.divide(worst_r[str(k+1)], r2vals)
                variables = [pvalues, logpvals, corrs, r2vals, \
                    indicators[str(k+1)], worst_p[str(k+1)], worst_r[str(k+1)],
                    p_ratio, r2_ratio]
                variable_names = ['pvalues', 'logpvals', 'correlations', \
                    'r2vals', 'indicators', 'worst_p', 'worst_r', 'p_ratio', 
                    'r2_ratio']

            elif statistic == 'ksc':
                r_ratio = np.divide(worst_r[str(k+1)], corrs)
                variables = [pvalues, logpvals, corrs, \
                    indicators[str(k+1)], worst_p[str(k+1)], worst_r[str(k+1)],
                    p_ratio, r_ratio]
                variable_names = ['pvalues', 'logpvals', 'correlations', \
                    'indicators', 'worst_p', 'worst_r', 'p_ratio', 'r_ratio']
            
        elif statistic == 'prop':
            prop_ratio = np.divide(worst_prop[str(k+1)], prop)
            variables = [prop, indicators[str(k+1)], prop_ratio, worst_prop[str(k+1)]]
            variable_names = ['prop','indicators','prop_ratio','worst_prop']

            # fxn to print var logx/y and var log x for all points and all pairs

        elif statistic == 'mine':
            variable_names = ['mine_str','mine_pvalues','worst_p','worst_str',\
                'nonlin','max_nonlin','min_nonlin','indicators']
            variables = mine_str, mine_pvalues, worst_mine_p, worst_mine_str, \
                mine_nonlin, max_mine_subset_nonlin, min_mine_subset_nonlin, indicators['1']
            # return variable_names
            # return variables


        statistics.report_results(n_var1, n_var2, working_dir, label,
                            initial_sig, true_sig, comb_to_rev, k+1)
        ###
        # Write R matrix
        ###

        output.print_Rmatrix(avg_var1, norm_avg_var1, avg_var2, norm_avg_var2, 
            var_var1, norm_var_var1, var_var2, norm_var_var2, skew_var1, 
            skew_var2, n_var1, n_var2, variable_names, variables, working_dir, 
            k+1, label, n_corr, statistic, paired)


    ### 
    # Examine Pointwise Statistics
    ###
    if point_compare or corr_compare and (statistic == 'kpc' or statistic == 'ksc'):
        statistics.pointwise_comparison(n_samp, n_var1, n_var2, samp_var1, 
            samp_var2, pvalues, corrs, working_dir, n_corr, threshold, 
            point_compare, corr_compare, statistic, paired, fold)

    print time.clock() - start_time
    return

if __name__ == "__main__":
    calculate_cutie()
