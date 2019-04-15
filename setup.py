#!/usr/bin/env python

from __future__ import division

from setuptools import setup
from glob import glob
import ast
import re

__author__ = "The Clemente Lab"
__copyright__ = "Copyright 2019, The Clemente Lab"
__credits__ = ["Jose C. Clemente, Kevin Bu"]
__license__ = "GPL"
__maintainer__ = "Kevin Bu"
__email__ = "kbu314@gmail.com"

# https://github.com/mitsuhiko/flask/blob/master/setup.py
_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('cutie/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(name='cutie',
      version=version,
      description='Correlations under the influence',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      url='http://github.com/clemente-lab/cutie',
      author=__author__,
      author_email=__email__,
      license=__license__,
      packages=['cutie'],
      scripts=glob('scripts/*py'),
      install_requires=[
          'click',
          'configparser',
          'numpy',
          'pandas',
          'statsmodels',
          'scipy',
          'matplotlib',
          'minepy',
          'seaborn',
          'py',
          'pytest'
      ],
      zip_safe=False)
