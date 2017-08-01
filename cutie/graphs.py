#!/usr/bin/env python
from __future__ import division
    
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

def generate_hist(
    pvalue_matrix, r2_matrix, 
    sig_points, label):
    ''' Function that takes in a matrix of a given statistic corresponding to (bact, meta) as well as a list of
    (bact, meta) points to plot, saved with a given statistic ('Pvalues') and label ('L6')
    '''
    pvalues = pvalue_matrix.flatten()
    fig = plt.figure()
    n_n, bins, patches = plt.hist(pvalues, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of All P-values', fontsize=20)
    plt.xlabel('P-values')
    plt.ylabel('Density')
    plt.axis([0, 1, 0, 50])
    fig.savefig('graphs/Histogram_All_PValues' + label + '.jpg')
    plt.close()
    
    sigpvalues = np.zeros(len(sig_points))
    k = 0
    for point in sig_points:
        i,j = point
        sigpvalues[k] = pvalue_matrix[i][j]
        k += 1
    
    fig = plt.figure()
    n_n, bins, patches = plt.hist(sigpvalues, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of All Significant P-values', fontsize=20)
    plt.xlabel('P-values')
    plt.ylabel('Density')
    plt.axis([0, 1, 0, 50])
    fig.savefig('graphs/Histogram of All Sig PValues' + label + '.jpg')
    plt.close()    
                
    logpvalues = [math.log(p) for p in sigpvalues]
    fig = plt.figure()
    n_n, bins, patches = plt.hist(logpvalues, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of All Significant Ln P-values', fontsize=20)
    plt.xlabel('Log P-values')
    plt.ylabel('Density')
    plt.axis([-42, -14, 0, 1])
    fig.savefig('graphs/Histogram of All Sig Log PValues' + label + '.jpg')
    plt.close()    

    r2values = r2_matrix.flatten()
    fig = plt.figure()
    n_n, bins, patches = plt.hist(r2values, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of R2-values', fontsize=20)
    plt.xlabel('R2-values')
    plt.ylabel('Density')
    plt.axis([0, 1, 0, 100])
    fig.savefig('graphs/Histogram of All R2Values' + label + '.jpg')
    plt.close()
    return 

def graph_insig(subj_id_list, initial_sig,
                subj_bact_dict, subj_meta_dict,
                metabolite_list, bact_list,
                label, stop = 100):
    ''' Function that plots the scatterplot for all (bact, meta) pairs that are not significant
    DO NOT CHANGE stop unless you want a lot of graphs...automatically stops at 100 by default
    '''
    all_points = list()
    for bact in xrange(0,len(bact_list)):
        for meta in xrange(0,len(metabolite_list)):
            point = (bact,meta)
            all_points.append(point)
    
    for point in initial_sig:
        all_points.remove(point)
    
    n = len(subj_bact_dict)
    counter = 0
    while counter < stop:
        bact_index,meta_index = all_points[counter]
        x = np.zeros(shape=n)
        y = np.zeros(shape=n)
        for subject in xrange(0,n): # sample
            subj_id = subj_id_list[subject]
            x[subject] = subj_bact_dict[subj_id][bact_index]
            y[subject] = subj_meta_dict[subj_id][meta_index]

        bacteria = bact_list[bact_index]
        metabolite = metabolite_list[meta_index]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        logp = math.log(p_value)
        
        fig = plt.figure()
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, slope*x + intercept, '-')
        fig.suptitle('(Bact, Meta)' + ' = ' + '(' + str(bact_index) + ', ' + str(meta_index) + ')' + ' R2 = ' + str('%.4f' % (r_value ** 2)) + ' ln p = ' + str('%.4f' % logp), fontsize=20)
        plt.xlabel('Bacteria ' + str(bacteria) + ' proportion', fontsize=18)
        plt.ylabel('Metabolite ' + str(metabolite) + ' level', fontsize=16)
        fig.savefig('graphs/insig/NonSignificant' + '_' + str(bact_index) + '_' + str(meta_index) + str(label) + '.jpg')
        plt.close()
        counter += 1
    return

def graph_initialsig(subj_id_list, initial_sig,
                     subj_bact_dict, subj_meta_dict,
                     bact_list, metabolite_list, 
                     label):
    ''' Function that plots the scatterplot for all (bact, meta) pairs that are initially significant (prior to resampling)
    '''
    n = len(subj_bact_dict)
    for point in initial_sig:
        bact_index,meta_index = point
        x = np.zeros(shape=n)
        y = np.zeros(shape=n)
        for subject in xrange(0,n): # sample
            subj_id = subj_id_list[subject]
            x[subject] = subj_bact_dict[subj_id][bact_index]
            y[subject] = subj_meta_dict[subj_id][meta_index]

        bacteria = bact_list[bact_index]
        metabolite = metabolite_list[meta_index]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        logp = math.log(p_value)

        fig = plt.figure()
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, slope*x + intercept, '-')
        fig.suptitle('(Bact, Meta)' + ' = ' + '(' + str(bact_index) + ', ' + str(meta_index) + ')' + ' R2 = ' + str('%.4f' % (r_value ** 2)) + ' ln p = ' + str('%.4f' % logp), fontsize=20)
        plt.xlabel('Bacteria' + str(bacteria) + 'proportion', fontsize=18)
        plt.ylabel('Metabolite ' + str(metabolite) + ' level', fontsize=16)
        fig.savefig('graphs/initialsig/Initial_Significant' + '_' + str(bact_index) + '_' + str(meta_index)+ str(label) + '.jpg')
        plt.close()
    return

def graph_truesig(subj_id_list, true_sig,
                  subj_bact_dict, subj_meta_dict,
                  bact_list, metabolite_list,
                  label):
    ''' Function that plots the scatterplot for all (bact, meta) pairs that remain significant following resampling
    '''
    n = len(subj_bact_dict)
    for point in true_sig:
        bact_index,meta_index = point
        x = np.zeros(shape=n)
        y = np.zeros(shape=n)
        for subject in xrange(0,n): # sample
            subj_id = subj_id_list[subject]
            x[subject] = subj_bact_dict[subj_id][bact_index]
            y[subject] = subj_meta_dict[subj_id][meta_index]
        bacteria = bact_list[bact_index]
        metabolite = metabolite_list[meta_index]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        logp = math.log(p_value)

        fig = plt.figure()
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, slope*x + intercept, '-')
        fig.suptitle('(Bact, Meta)' + ' = ' + '(' + str(bact_index) + ', ' + str(meta_index) + ')' + ' R2 = ' + str('%.4f' % (r_value ** 2)) + ' ln p = ' + str('%.4f' % logp), fontsize=20)
        plt.xlabel('Bacteria' + str(bacteria) + 'proportion', fontsize=18)
        plt.ylabel('Metabolite ' + str(metabolite) + ' level', fontsize=16)
        fig.savefig('graphs/truesig/Significant' + '_' + str(bact_index) + '_' + str(meta_index)+ '.jpg')
        plt.close()
    return 

def initial_stats_MLR(
    subj_meta_dict, subj_bact_dict,
    subj_id_list, predictors,
    threshold= 0.05):
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

def resample_MLR(meta_index, subj_meta_dict,
                 subj_bact_dict, subj_id_list,
                 predictors, threshold): 
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

def update_stats_MLR(subj_id_list,subj_bact_dict,
                     subj_meta_dict, predictors, 
                     initial_sig, MLR_pvalue_dict, 
                     threshold = 0.05, levels = [0.05,0.01,0.001,0.0001,0.00001]):
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

def generate_bact_hist(pvalue_matrix, r2_matrix, 
                       corr_matrix, label):
    ''' Function that takes in a matrix of a given statistic corresponding to (bact, meta) as well as a list of
    (bact, meta) points to plot, saved with a given statistic ('Pvalues') and label ('L6')
    '''
    pvalues = pvalue_matrix.flatten()
    fig = plt.figure()
    n_n, bins, patches = plt.hist(pvalues, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of All P-values', fontsize=20)
    plt.xlabel('P-values')
    plt.ylabel('Density')
    plt.axis([0, 1, 0, 50])
    fig.savefig('graphs/Histogram_All_PValues' + label + '.jpg')
    plt.close()
    
    logpvalues = [math.log(p) for p in pvalues]
    fig = plt.figure()
    n_n, bins, patches = plt.hist(logpvalues, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of All Ln P-values', fontsize=20)
    plt.xlabel('Log P-values')
    plt.ylabel('Density')
    plt.axis([-42, -14, 0, 1])
    fig.savefig('graphs/Histogram_All_LogPValues' + label + '.jpg')
    plt.close()    

    r2values = r2_matrix.flatten()
    fig = plt.figure()
    n_n, bins, patches = plt.hist(r2values, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of R2-values', fontsize=20)
    plt.xlabel('R2-values')
    plt.ylabel('Density')
    plt.axis([0, 1, 0, 100])
    fig.savefig('graphs/Histogram_All_R2Values' + label + '.jpg')
    plt.close()

    corrvalues = corr_matrix.flatten()
    corrvalues = corrvalues[~np.isnan(corrvalues)]
    fig = plt.figure()
    n_n, bins, patches = plt.hist(corrvalues, 100, normed=1, facecolor='green', alpha=0.75)
    fig.suptitle('Histogram of Corr-values', fontsize=20)
    plt.xlabel('Corr-values')
    plt.ylabel('Density')
    plt.axis([0, 1, 0, 100])
    fig.savefig('graphs/Histogram_All_CorrValues' + label + '.jpg')
    plt.close()

    return