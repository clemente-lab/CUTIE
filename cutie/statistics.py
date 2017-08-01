#!/usr/bin/env python
from __future__ import division
    
import os
import numpy as np
from scipy import stats
import statsmodels.api as sm

def initial_stats_SLR(
        subj_id_list, subj_bact_dict, 
        subj_meta_dict, label):
    """ Function that computes an initial matrix of simple linear regression (SLR) correlations, R2 values and p-values 
    between each metabolite and bacteria. Returns a dictionary of m x t matrices where the key is the statistic of interest
    and the element is a matrix of values for the keyed statistic (correlation coefficient, R2 value and pvalue)  
    where each entry of the matrix corresponds to the value for the SLR between bacteria i and metabolite j
    label = 'L6'
    label = str(subj_bact_ns_fp).split('_')[-1].split('.')[0]
    """
    t = len(subj_meta_dict[subj_meta_dict.keys()[0]])
    m = len(subj_bact_dict[subj_bact_dict.keys()[0]])
    n = len(subj_id_list)
    # print t,m,n
    
    # define statistics of interest,
    # matrices to hold the values of each statistic, 
    # created, a dict containing an element=indicator variable if that file corresponding to key=statistic will be created 
    # and fnames, a dict of element=file name corresponding to each key=statistic
    statistics = ['corr', 'pvalue', 'r2']
    matrices = dict()
    created = dict()
    fnames = dict()

    # if file already exists, read them in
    for statistic in statistics:
        fnames[statistic]= 'data_processing/initial_' + statistic + '_' + str(label) + '_matrix.txt'
        created[statistic] = True # assumes file will be created unless shown otherwise
        matrices[statistic] = np.zeros(shape=(m,t))
    
        if os.path.isfile(fnames[statistic]) is True:
            print 'file ' + str(fnames[statistic]) +  ' already exists'
            stat_file = open(fnames[statistic],'r')
            split_lines = stat_file.read().split('\n')
            split_lines.pop() # remove last row which is empty
            for i in xrange(0,len(split_lines)):
                lines = split_lines[i].split('\t') 
                lines.pop() # remove the space after the last tab
                for j in xrange(0,len(lines)):
                    try:
                        matrices[statistic][i][j] = np.float(lines[j])
                    except:
                        matrices[statistic][i][j] = np.ma.masked
            created[statistic] = False
            stat_file.close()
    
    # compute statistics and create file to write them in
    for statistic in statistics:
        if created[statistic] is True:
            print 'new file ' + str(fnames[statistic]) + ' created'
            stat_file = open(fnames[statistic],'w')
            
            for i in xrange(0,m): # bacteria
                for j in xrange(0,t): # metabolite
                    bact = np.zeros(shape=n)
                    meta = np.zeros(shape=n)
                    for k in xrange(0,n): # sample
                        subj_id = subj_id_list[k]
                        bact[k] = subj_bact_dict[subj_id][i] 
                        meta[k] = subj_meta_dict[subj_id][j]
                    stat_value = np.float(0.0)
                    if statistic == 'corr':
                        stat_value = np.ma.corrcoef(bact,meta)[0,1]
                        matrices[statistic][i][j] = stat_value
                    else:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(bact,meta)
                        if statistic == 'pvalue':
                            stat_value = p_value
                            matrices[statistic][i][j] = stat_value
                        if statistic == 'r2':
                            stat_value = r_value ** 2
                            matrices[statistic][i][j] = stat_value
                    stat_file.write(str(stat_value) + '\t')           
                stat_file.write('\n')
            stat_file.close()

    return matrices

def resample_SLR(bacteria_index, metabolite_index, 
                 subj_id_list, subj_bact_dict, 
                 subj_meta_dict):
    """ Function that takes a given bacteria and metabolite by index and recomputes correlation 
    by removing 1 out of n (sample_size) points from subj_id_list. Returns an np.array of pvalues where pvalue_list[i] 
    corresponds to the pvalue of the correlation after removing the sample of subj_id_list[i]
    """
    sample_size = len(subj_bact_dict)
    pvalue_list = np.zeros(sample_size)
    bact = np.zeros(sample_size)
    meta = np.zeros(sample_size)
    for sample_index in xrange(0,sample_size): 
        subj_id = subj_id_list[sample_index]
        bact[sample_index] = subj_bact_dict[subj_id][bacteria_index]
        meta[sample_index] = subj_meta_dict[subj_id][metabolite_index]
   
    # iteratively delete one sample
    for sample_index in xrange(0,sample_size):
        new_bact = np.ndarray.copy(bact)
        new_meta = np.ndarray.copy(meta)
        new_bact = np.delete(new_bact, sample_index)
        new_meta = np.delete(new_meta, sample_index)
        slope, intercept, r_value, p_value, std_err = stats.linregress(new_bact,new_meta)
        pvalue_list[sample_index] = p_value
            
    return pvalue_list

def update_stats_SLR(
        subj_id_list, subj_bact_dict,
        subj_meta_dict, pvalue_matrix, 
        correlation_matrix, levels = [0.05,0.01,0.001,0.0001,0.00001]):
    ''' Function that recomputes pvalues by dropping 1 different observation at a time 
    Returns an array of p-values where each i-th entry corresponds to the recomputed p-value 
    of the correlation after dropping observation i
    '''
    # pvalue_matrix MUST correspond with correlation_matrix entries
    n_meta = len(subj_meta_dict[subj_meta_dict.keys()[0]])
    n_bact = len(subj_bact_dict[subj_bact_dict.keys()[0]])
    n_subj = len(subj_id_list)

    threshold = 0.05 / (n_meta * n_bact)
    total_entries = np.count_nonzero(correlation_matrix)
    non_zero_entries = np.count_nonzero(~np.isnan(correlation_matrix))
    
    pvalue_dist = dict()
    #print 'The total number of correlations is ' + str(total_entries)
    #print 'The total number of non-zero correlations is ' + str(non_zero_entries)
    #print 'The threshold used was ' + str(threshold)
    for l in xrange(0,len(levels)): 
        count = ((pvalue_matrix < levels[l]) & (pvalue_matrix != 0)).sum()
        pvalue_dist[levels[l]] = count
        #print 'The number of correlations with p-value less than ' + str(levels[l]) + ' is ' + str(count)
    
    initial_sig = list()
    true_sig = list()
    indicator_sig = np.zeros(shape = (n_bact,n_meta))
    indicator_insig = np.zeros(shape = (n_bact,n_meta))
    for bact in xrange(0,n_bact): # bacteria
        for meta in xrange(0,n_meta): # metabolite
            point = (bact,meta)
            if pvalue_matrix[bact][meta] < threshold:
                indicator_sig[bact][meta] = 1
                initial_sig.append(point)
                pvalue_list = resample_SLR(bact, meta, subj_id_list, subj_bact_dict, subj_meta_dict) # is really an np array
                # count number of entries above threshold 
                highp = pvalue_list[np.where( pvalue_list > threshold ) ]
                # indicator if the correlation no longer is significant
                if np.sum(highp) > 0:
                    indicator_insig[bact][meta] = 1
                else:
                    true_sig.append(point)
    ntotalsig = np.sum(indicator_sig)
    nfalsesig = np.sum(indicator_sig * indicator_insig)
    ntruesig = np.sum(indicator_sig) - np.sum(indicator_sig * indicator_insig)
    
    return pvalue_dist, ntotalsig, nfalsesig, ntruesig, initial_sig, true_sig

def initial_stats_MLR(subj_meta_dict,subj_bact_dict,subj_id_list,predictors,threshold= 0.05):
    '''Function that performs multiple linear regression using a subset of bacterial predictors
    and returns
        (1) a dict of initial p-values with key=metabolite index and element=p-value array 
        corresponding to each predictor and metabolite pair (the 0th index of this array 
        corresponds to the coefficient p-value)
        (2) a list of bacteria, metabolite pairs corresponding to significant coefficients
    '''
    n_meta = len(subj_meta_dict[subj_meta_dict.keys()[0]])
    n_bact = len(subj_bact_dict[subj_bact_dict.keys()[0]]) 
    n_subj = len(subj_id_list) 
    n_pred = len(predictors)
    
    MLR_pvalue_dict = dict()
    initial_sig = list()
    
    # for each metabolite we conduct a MLR with the main predictors found from SLR
    for meta in xrange(0,n_meta):
        outcome_array = np.zeros(n_subj)
        predict_array = np.zeros(shape=(n_subj,n_pred+1)) # +1 because of the intercept estimator beta_0
        
        for subj in xrange(0,n_subj):
            subj_id = subj_id_list[subj]
            bact_levels = subj_bact_dict[subj_id]
            
            for pred in xrange(0,n_pred):
                predict_array[subj][pred+1] = bact_levels[predictors[pred]]
        
            outcome_array[subj] = subj_meta_dict[subj_id][meta]

        results = sm.OLS(outcome_array, predict_array).fit()
        MLR_pvalue_dict[meta] = results.pvalues
        
        for pvalue in xrange(1,n_pred+1):
            if results.pvalues[pvalue] < threshold:
                point = predictors[pvalue-1],meta
                initial_sig.append(point)
        
            
    return initial_sig, MLR_pvalue_dict

def resample_MLR(meta_index, subj_meta_dict,subj_bact_dict,subj_id_list,predictors,threshold): 
    ''' Function that takes a given metabolite and recomputes MLR pvalues for each coefficient
        by removing all samples one each time, returning an array of dimensions (sample x (predictors)) 
        We ignore the beta_0 predictor
    '''
    n_meta = len(subj_meta_dict[subj_meta_dict.keys()[0]])
    n_bact = len(subj_bact_dict[subj_bact_dict.keys()[0]]) 
    n_subj = len(subj_id_list) 
    n_pred = len(predictors)
    
    pvalue_array = np.zeros(shape=(n_subj,n_pred))
    threshold = 0.05

    # for each metabolite we conduct a MLR with the main predictors found from SLR
    outcome_array = np.zeros(n_subj)
    predict_array = np.zeros(shape=(n_subj,n_pred+1)) # +1 because of the intercept estimator beta_0

    # populate the pred matrix and outcome matrix
    for subj in xrange(0,n_subj):
        subj_id = subj_id_list[subj]
        bact_levels = subj_bact_dict[subj_id]

        for pred in xrange(0,n_pred):
            predict_array[subj][pred+1] = bact_levels[predictors[pred]]

        outcome_array[subj] = subj_meta_dict[subj_id][meta_index]
    
    for subj in xrange(0,n_subj):
        new_outcome = np.ndarray.copy(outcome_array)
        new_predict = np.ndarray.copy(predict_array)
        new_outcome = np.delete(new_outcome, subj, 0)
        new_predict = np.delete(new_predict, subj, 0)
        
        results = sm.OLS(new_outcome, new_predict).fit()
        pvalue_array[subj] = results.pvalues[1:len(results.pvalues)]
        
    return pvalue_array

def update_stats_MLR(subj_id_list,subj_bact_dict,subj_meta_dict, predictors, initial_sig, MLR_pvalue_dict, threshold = 0.05, levels = [0.05,0.01,0.001,0.0001,0.00001]):
    ''' Function that recomputes pvalues by dropping 1 different observation at a time 
    Returns 
    (1) a dictionary of pvalues where
        key = metabolite number
        element = np.array of p-values (dimensions n_subj x n_pred) where each point i,j in the np.array 
        corresponds to the p-value corresponding to the predictor j when subject i has been removed
    (2) true_sig, a list of points (bact,meta) that are still significant
    '''
    # pvalue_matrix MUST correspond with correlation_matrix entries
    n_meta = len(subj_meta_dict[subj_meta_dict.keys()[0]])
    n_bact = len(subj_bact_dict[subj_bact_dict.keys()[0]])
    n_subj = len(subj_id_list)
    n_pred = len(predictors)

    total_entries = n_meta * n_pred
    
    pvalue_dist = dict()
    print 'The total number of coefficients is ' + str(total_entries)
    print 'The threshold used was ' + str(threshold)
    for meta in xrange(0,len(subj_meta_dict[subj_meta_dict.keys()[0]])):
        pvalue_list = MLR_pvalue_dict[meta][1:len(MLR_pvalue_dict[meta])]
        for l in xrange(0,len(levels)): 
            count = (( pvalue_list < levels[l]) & (pvalue_list != 0)).sum()
            pvalue_dist[levels[l]] = count
            # print 'For metabolite ' + str(meta) + ', the number of predictors with p-value less than ' + str(levels[l]) + ' is ' + str(count)
    
    # each point (i,j) corresponds to (bact, meta)
    true_sig = list()
    indicator_sig = np.zeros(shape = (n_pred,n_meta))
    indicator_insig = np.zeros(shape = (n_pred,n_meta))
    for meta in xrange(0, n_meta):
        pvalue_list = MLR_pvalue_dict[meta][1:len(MLR_pvalue_dict[meta])]
        for pred in xrange(0,n_pred):
            point = (pred,meta)
            if MLR_pvalue_dict[meta][pred+1] < threshold:
                indicator_sig[pred][meta] = 1
                pvalue_array = resample_MLR(meta, subj_meta_dict,subj_bact_dict, subj_id_list, predictors,threshold) # is really an np array
                highp = pvalue_list[np.where( pvalue_list > threshold ) ]
                # indicator if the correlation no longer is significant
                if np.sum(highp) > 0:
                    indicator_insig[pred][meta] = 1
                else:
                    true_sig.append(point)
    ntotalsig = np.sum(indicator_sig)
    nfalsesig = np.sum(indicator_sig * indicator_insig)
    ntruesig = np.sum(indicator_sig) - np.sum(indicator_sig * indicator_insig)
    
    return pvalue_dist, ntotalsig, nfalsesig, ntruesig, true_sig

def bact_corr(subj_id_list, 
              subj_bact_dict, 
              threshold = 0.05):
    ''' Function that takes in bacteria levels in subj_bact_dict and a subj_id_list (list of subjects)
    and returns matrices corresponding to the correlation, R2, and pvalue of each pairwise correlation
    '''
    n_bact = len(subj_bact_dict[subj_bact_dict.keys()[0]])
    n_subj = len(subj_id_list)
    
    bact_corr_matrix = np.zeros(shape=(n_bact,n_bact))
    bact_r2_matrix = np.zeros(shape=(n_bact,n_bact))
    bact_pvalue_matrix = np.zeros(shape=(n_bact,n_bact))

    bactx = np.zeros(n_subj)
    bacty = np.zeros(n_subj)

    for bacti in xrange(0,n_bact):
        for bactj in xrange(0,n_bact):
            for subj in xrange(0,n_subj):
                subj_id = subj_id_list[subj]
                bactx[subj] = subj_bact_dict[subj_id][bacti]  
                bacty[subj] = subj_bact_dict[subj_id][bactj]

            corr = np.ma.corrcoef(bactx,bacty)[0,1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(bactx,bacty)

            bact_corr_matrix[bacti][bactj] = corr
            bact_r2_matrix[bacti][bactj] = r_value **2 
            bact_pvalue_matrix[bacti][bactj] = p_value

    #subtract diagonals and divide by 2
    print 'The total number of correlations is ' + str(n_bact * (n_bact - 1)/2)
    print 'The number of correlations with corr > 0.99 is ' + str((((bact_corr_matrix > 0.99)).sum() - 897)/2)
    print 'The number of correlations with r2 > 0.9 is ' + str((((bact_r2_matrix > 0.9)).sum() - 897)/2) 
    print 'The number of correlations with pvalue < ' + str(threshold) + ' is ' + str((((bact_pvalue_matrix < threshold)).sum() - 897)/2)
    return bact_corr_matrix, bact_r2_matrix, bact_pvalue_matrix