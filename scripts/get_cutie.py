#!/usr/bin/env python
from __future__ import division

import click
from cutie import parse
from cutie import __version__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)

@click.option('-i', '--input_fp', required=True,
              type=click.Path(exists=True),
              help='Input  file')

def get_cutie(input_fp):
    """ Read and parse input file.
    """
    data = parse.example(open(input_fp,'r'))
    
    for d in data:
        print d, data[d]
        
if __name__ == "__main__":
    get_cutie()
