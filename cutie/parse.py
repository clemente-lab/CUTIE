#!/usr/bin/env python
from __future__ import division

import re
import sys

def example(in_f, sep='\t', body_sep=',', comment_sep='#'):
    """ Example parsing function.
    Returns a dictionary with key=first element of each line; values=list with rest of elements
    """
    ret_values = {}

    for line in in_f:
        line = line.strip()
        # Skip comment lines
        if line and not line.startswith(comment_sep):
            head, body = line.split(sep)
            body_parts = body.split(body_sep)
            # Ensure unique keys
            if head not in ret_values:
                ret_values[head] = body_parts
            else:
                print 'ERR: Duplicated entry: [', head, ']'
                sys.exit(-1)

    return ret_values
