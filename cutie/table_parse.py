#!/usr/bin/env python
from __future__ import division
    
def subject_table_to_dict(input_fp):
    """ Function that reads in a file from path where the start of each line is the subject id and the subsequent
    tab-delimited values are either metabolite or bacteria levels and loads it into a python dictionary where the key
    is the subject id and each element is a list of values corresponding to that subject id
    """
    subject_dict = dict()
    fname = open(input_fp,'r')
    split_lines = fname.read().split('\n')
    split_lines.pop() # remember to pop off the last row
    for line in split_lines:
        line = line.split('\t')
        key = line[0] 
        subject_dict[key] = line[1:len(line)] 
    return subject_dict