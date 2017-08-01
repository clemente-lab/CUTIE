#!/usr/bin/env python
from __future__ import division

import click
from cutie import data_parse
from cutie import table_parse
from cutie import statistics
from cutie import graphs
from cutie import __version__
import os

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)

@click.option('-sm', '--subj_meta_fp', required=True,
              type=click.Path(exists=True),
              help='Input  subj meta file')
@click.option('-ns', '--subj_bact_ns_fp', required=True,
              type=click.Path(exists=True),
              help='Input  nonsmoking file')
@click.option('-s', '--subj_bact_s_fp', required=True,
              type=click.Path(exists=True),
              help='Input  smoking file')

def stats_cutie(subj_meta_fp, 
                subj_bact_ns_fp, 
                subj_bact_s_fp):
    """ Read in text subj_meta and subj_bact files and generate statistics
    subj_meta_fp = 'data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt'
    subj_bact_ns_fp = 'data/otu_table_MultiO__Status_Smoker_Non.Smoker___L6.txt'
    subj_bact_s_fp = 'data/otu_table_MultiO__Status_Smoker_Smoker___L6.txt'
    """
    ### 
    # Pre-processing
    ###
    
    # create subfolder to hold data analysis files
    if os.path.exists('data_processing') is not True:
        os.makedirs('data_processing')
        print 'Created directory /data_processing'
    else:
        print 'Directory /data_processing already exists'
    
    subj_id_list, metabolite_list = data_parse.subj_meta_parse(subj_meta_fp, 
                                                    'data_processing/subj_meta_table.txt')
    
    # special case/hard coded modifications
    exceptions = ['110705RC.N.1.RL']
    for exception in exceptions:
        subj_id_list.remove(exception)
        print 'Removed subject ID ' + str(exception)
    #print 'The length of subj_id_list is ' + str(len(subj_id_list))
    #print 'The length of metabolite_list is ' + str(len(metabolite_list))
    
    # merge the non-smoking and smoking samples and outputs table 'subj_bact_table_L6.txt'
    label = str(subj_bact_ns_fp).split('_')[-1].split('.')[0]
    newfile_fp = 'data_processing/subj_bact_table_' + str(label) + '.txt'
    bact_list = data_parse.subj_bact_merge_parse(subj_bact_ns_fp,
                                      subj_bact_s_fp,
                                      newfile_fp)
    # print 'The length of bact_list is ' + str(len(bact_list))
    
    subj_meta_dict = table_parse.subject_table_to_dict('data_processing/subj_meta_table.txt')
    subj_bact_dict = table_parse.subject_table_to_dict(newfile_fp)
    
    for exception in exceptions:
        subj_meta_dict.pop(exception, None)
        subj_bact_dict.pop(exception, None)
    
    #print 'The length of subj_meta_dict is ' + str(len(subj_meta_dict))
    #print 'The length of subj_bact_dict is ' + str(len(subj_bact_dict))
    
    ###
    # Simple Linear Regression and Pairwise correlations
    ### 
    
    matrices = statistics.initial_stats_SLR(subj_id_list, subj_bact_dict, 
                                            subj_meta_dict, label)
    
    """for key in matrices.keys():
        print matrices[key][20][20]
    """
    corr_matrix = matrices['corr'] 
    pvalue_matrix = matrices['pvalue'] 
    r2_matrix = matrices['r2']

    levels =[0.05,0.01,0.001,0.0001,0.00001]
    SLR_pvalue_dist, SLR_ntotalsig, SLR_nfalsesig, SLR_ntruesig, SLR_initial_sig, SLR_true_sig = statistics.update_stats_SLR(subj_id_list, subj_bact_dict,
                                                                                                                  subj_meta_dict, pvalue_matrix,
                                                                                                                  corr_matrix,levels)
    # print 'The length of SLR_pvalue_dist is ' + str(len(SLR_pvalue_dist))
    print 'The number of total, false, and true significant corr is ' + str(SLR_ntotalsig) + ', ' + str(SLR_nfalsesig) + ', ' + str(SLR_ntruesig) + ', respectively'
    print 'The number of initially significant corr and true corr is ' + str(len(SLR_initial_sig)) + ' and ' + str(len(SLR_true_sig)) 

    ###
    # Graph Generation
    ### 
    
    # create subfolder to hold graphs 
    if os.path.exists('graphs') is not True:
        os.makedirs('graphs')
        print 'Created directory /graphs'

    
    graphs.generate_hist(pvalue_matrix, r2_matrix, 
                         SLR_initial_sig, label)
    
    if os.path.exists('graphs/insig') is not True:
        os.makedirs('graphs/insig')
        print 'Created directory /graphs/insig'
        
    graphs.graph_insig(subj_id_list, SLR_initial_sig,
                subj_bact_dict, subj_meta_dict,
                metabolite_list, bact_list,
                label, stop = 100)
    
    if os.path.exists('graphs/initialsig') is not True:
        os.makedirs('graphs/initialsig')
        print 'Created directory /graphs/initialsig'
    
    graphs.graph_initialsig(subj_id_list, SLR_initial_sig,
                            subj_bact_dict, subj_meta_dict,
                            bact_list, metabolite_list, 
                            label)
    
    if os.path.exists('graphs/truesig') is not True:
        os.makedirs('graphs/truesig')
        print 'Created directory /graphs/truesig'
        
    graphs.graph_truesig(subj_id_list, SLR_true_sig,
                        subj_bact_dict, subj_meta_dict,
                        bact_list, metabolite_list, label)
    
    print 'Generated histograms and scatterplots'
    
    ###
    # Multiple Linear Regression
    ###
    
    predictors = list()
    # print 'The SLR true_sig bact,meta pairs are ' + str(SLR_true_sig)
    for point in SLR_true_sig:
        bact, meta = point
        if bact not in predictors:
            predictors.append(bact)
    print 'The MLR predictors are ' + str(predictors)
    print 'The number of MLR predictors is ' + str(len(predictors))
    
    MLR_threshold = 0.05
    
    MLR_initial_sig, MLR_pvalue_dict = statistics.initial_stats_MLR(subj_meta_dict, subj_bact_dict,
                                                                    subj_id_list, predictors, 
                                                                    MLR_threshold)
    
    MLR_pvalue_dist, MLR_ntotalsig, MLR_nfalsesig, MLR_ntruesig, MLR_true_sig = statistics.update_stats_MLR(subj_id_list, subj_bact_dict,
                                                                                                            subj_meta_dict, predictors, 
                                                                                                            MLR_initial_sig, MLR_pvalue_dict, 
                                                                                                            MLR_threshold, levels=[0.05])
    
    print 'There are ' + str(len(MLR_initial_sig)) + ' MLR_initial_sig which are ' + str(MLR_initial_sig)
    overlap = set(SLR_true_sig).intersection(MLR_initial_sig)
    print 'The overlap between SLR true_sig and MLR initial_sig is ' + str(len(overlap)) 
    print 'There are ' + str(len(MLR_true_sig)) + ' MLR_true_sig' # + 'which are ' + str(MLR_true_sig)
    
    ###
    # Correlation between bacteria predictors
    ###
    # will take some time to run
    """bact_corr_matrix, bact_r2_matrix, bact_pvalue_matrix = statistics.bact_corr(subj_id_list, subj_bact_dict, threshold = 0.05)
                graphs.generate_bact_hist(bact_pvalue_matrix, bact_r2_matrix, 
                                          bact_corr_matrix, label + 'MLR_corr')
                """
if __name__ == "__main__":
    stats_cutie()
