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
    plt.axis([-50, 0, 0, 1])
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