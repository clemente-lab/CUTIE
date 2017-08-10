#!/usr/bin/env python
from __future__ import division

import click
from cutie import parse
from cutie import statistics
from cutie import __version__
import os

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)

@click.option('-sm', '--samp_meta_fp', required=True,
              type=click.Path(exists=True),
              help='Input  samp meta file')
@click.option('-sb', '--samp_bact_fp', required=True,
              type=click.Path(exists=True),
              help='Input  samp bact file')

def calculate_SLR(samp_meta_fp, 
                  samp_bact_fp,
                  working_dir = ''):
    """ 
    INPUTS
    samp_meta_fp: file object pointing to samp_meta data file
    samp_bact_fp: file object pointing to samp_bact data file (OTU table)
    working_dir:  file path to where data is processed
    
    FUNCTION
    Computes pairwise correlations between each bact, meta pair and 
    for the significant correlations, recomputes correlation for each pair
    after iteratively excluding individual observations, differentiating
    true and false correlations on the basis of whether the correlation remains
    significant when each individual observation is dropped
    
    EXAMPLE
    stats_cutie('data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt',
            'data/otu_table_MultiO_merged___L6.txt')
    """
    
    ### 
    # Parsing and Pre-processing
    ###
    
    # create subfolder to hold data analysis files
    if os.path.exists(working_dir + 'data_processing') is not True:
        os.makedirs(working_dir + 'data_processing')
        
    # parse sample-metabolite data table
    # meta file doesn't split on \n but will on \r 
    with open(samp_meta_fp,'rU') as f:    
        samp_ids, meta_names, samp_meta_dict = parse.samp_meta_parse(f)
    
    # extract 'L6' or 'L7' label from OTU tables
    label = str(samp_bact_fp).split('_')[-1].split('.')[0] 
    
    # parse OTU table
    with open(samp_bact_fp, 'r') as f:
        bact_names, samp_bact_dict = parse.samp_bact_parse(f)
    
    # special case/hard coded modifications
    exceptions = ['110705RC.N.1.RL']
    for exception in exceptions:
        samp_ids.remove(exception)
        samp_meta_dict.pop(exception, None)
        samp_bact_dict.pop(exception, None)
        print 'Removed subject ID ' + str(exception)
    
    print 'The length of samp_ids is ' + str(len(samp_ids))
    print 'The length of metabolite_names is ' + str(len(meta_names))
    print 'The length of bact_names is ' + str(len(bact_names))

    samp_bact_matrix = parse.dict_to_matrix(samp_bact_dict, samp_ids)
    samp_meta_matrix = parse.dict_to_matrix(samp_meta_dict, samp_ids)
    
    ###
    # Simple Linear Regression and Pairwise correlations
    ### 
    
    stat_dict = statistics.initial_stats_SLR(samp_ids, 
                                             samp_bact_matrix, 
                                             samp_meta_matrix, 
                                             working_dir,
                                             label)

    '''
    For key = 'stats.linregress', the statistics are:
                   ['b1', 'b0', 'pcorr','pvalue','r2']
                   For key = 'stats.spearmanr', the statistics are:
                   ['scorr','pvalue']
    '''
    pvalue_matrix = stat_dict['stats.linregress'][3]
    
    SLR_initial_sig, SLR_true_sig = statistics.update_stats_SLR(samp_ids, 
                                                                samp_bact_matrix,
                                                                samp_meta_matrix, 
                                                                pvalue_matrix)
    
    print 'The number of false correlations is ' + str(len(SLR_initial_sig)) 
    print 'The number of true correlations is ' + str(len(SLR_true_sig))
    
    return

if __name__ == "__main__":
    calculate_SLR()
