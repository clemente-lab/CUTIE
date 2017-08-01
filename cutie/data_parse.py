#!/usr/bin/env python
from __future__ import division
    
import os
def subj_meta_parse (sm_file_path, fname):
    """ Function that reads in a subject and metabolite (sm) data file and creates a table with name fname where each 
    row begins with a subject id and each subsequent tab-delimited value is the level of metabolite. Returns a list of subject ids
    (subj_id_list) and a list of metabolites (metabolite_list)
    ex.
    sm_file_path = 'data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt'
    subj_id_list, metabolite_list = sm_file_to_table(sm_file_path,'subj_meta_table.txt')
    """
    sample_m_file = open(sm_file_path,'r')
    subj_id_list = list()
    
    # create subject-metabolite table file to write into
    created = False
    if os.path.isfile(fname) is True:
        print 'file ' + str(fname) + ' already exists'
        sm_table_file = open(fname,'r')
    else: 
        print 'new file ' + str(fname) + ' created'
        sm_table_file = open(fname,'w') 
        created = True
    
    # Parse file for subj_id_list and metabolite_list
    split_lines = sample_m_file.read().split('\r')

    # generate metabolite list (M_1 to M_t); excludes 100th entry; metabolites are in col 17 to 99
    metabolite_list = split_lines[0].split('\t')[17:100] 
    split_lines.pop(0)
    for line in split_lines:
        line = line.split('\t')
        subj_id = line[0] #sample_id is the 0th entry not subj_id
        subj_id_list.append(subj_id)
        metabolites = line[17:100]
        if created is True:
            sm_table_file.write(subj_id)
            for metabolite in metabolites:
                sm_table_file.write('\t' + metabolite)
            sm_table_file.write('\n') # in the future you'll have to pop off the last line

    sm_table_file.close() 
    sample_m_file.close()
   
    return subj_id_list, metabolite_list

def subj_bact_merge_parse(
        subj_bact_ns_fp, 
        subj_bact_s_fp, 
        fname):
    """ Function that reads in both non-smoking and smoking subject-bacteria files and merges them into a single output
    table with name fname and each row begins with a subject id and each subsequent tab-delimited value is the level of bacteria 
    ex.
    sb_ns_L6_path = 'data/otu_table_MultiO__Status_Smoker_Non.Smoker___L6.txt'
    sb_s_L6_path = 'data/otu_table_MultiO__Status_Smoker_Smoker___L6.txt'
    bact_list_L6 = subj_bact_merge_parse(sb_ns_L6_path,sb_s_L6_path,'sb_L6_table.txt')
    """
    # create files
    sb_ns_file = open(subj_bact_ns_fp,'r')
    sb_s_file = open(subj_bact_s_fp,'r')
    sb_files = [sb_ns_file, sb_s_file]
    
    # create temporary lists (corresponding to smoking and non-smoking files) to be subsequently merged
    bact_list_ns = list()
    bact_list_s = list()
    bact_lists = [bact_list_ns, bact_list_s]
    subj_ids_ns = list()
    subj_ids_s = list()
    subj_ids = [subj_ids_ns,subj_ids_s]
    
    # create subject bacteria table file to write into
    if os.path.isfile(fname) is True:
        print 'file ' + str(fname) + ' already exists'
        for i in xrange(0,len(sb_files)):
            split_lines = sb_files[i].read().split('\n')
            split_lines.pop(0) 
            split_lines.pop(0)
            split_lines.pop()
            for line in split_lines:
                bact_lists[i].append(line.split('\t')[0])
    else:
        print 'new file ' + str(fname) + ' created'
        subj_bact_table_file = open(fname,'w')
        for i in xrange(0,len(sb_files)):
            split_lines = sb_files[i].read().split('\n')
            split_lines.pop(0) 
            subj_ids[i] = split_lines[0].split('\t')
            subj_ids[i].pop(0) 
            split_lines.pop(0)
            split_lines.pop()
            for line in split_lines:
                bact_lists[i].append(line.split('\t')[0])
            for s in xrange(0,len(subj_ids[i])):
                subj_bact_table_file.write(subj_ids[i][s])
                for line in split_lines:
                    line = line.split('\t')
                    subj_bact_table_file.write('\t' + line[s+1])
                subj_bact_table_file.write('\n')
        subj_bact_table_file.close()
    
    sb_ns_file.close()
    sb_s_file.close()
    
    # if subject sets overlap in the smoking and nonsmoking files, print error
    if bool(set(subj_ids_ns) & set(subj_ids_s)) is not False:
        print 'your subject ids in the smoking and nonsmoking files overlap'
    
    # if the bacteria in both lists differ, print error
    if bool(set(bact_list_ns).difference(set(bact_list_s))) is not False:
        print 'the nonsmoking and smoking files contain different bacteria'
        print set(bact_list_ns).difference(set(bact_list_s))
    
    # otherwise return bacteria_list_ns
    bact_list = bact_list_ns
        
    return bact_list

