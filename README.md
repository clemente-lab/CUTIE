###
# Overview
###

CUTIE is a python module for evaluating pairwise correlations in a dataset for
influential points using various methods, including CUTIE, jackknifing, and
bootstrapping.

Detailed information about the package can be found in the accompanying publication.

Questions and concerns should be sent to kbu314@gmail.com.

###
# Installation, setup, and preprocessing
###

We suggest using CUTIE in a conda environment, as such.

source activate cutie

Perform installation by navigating to the main directory. Make sure you have
your PYTHONPATH variable set to the desired destination.

cd /path/to/CUTIE
export PYTHONPATH=/path/to/pythonpath
python3 setup.py install --prefix=/yourpath/

Likely you will need to install minepy unless you have used it before; it is
located at the following URL.
http://minepy.sourceforge.net/docs/0.3.5/install.html

###
# Usage example
###

The config file test_config.ini will need to be modified depending on where you
are running CUTIE.

mkdir lungpt_1pc_point_unit_test0.05/

python3 /Users/KevinBu/Desktop/clemente_lab/CUTIE/scripts/calculate_cutie.py -df /Users/KevinBu/Desktop/clemente_lab/CUTIE/tests/config_defaults.ini -cf /Users/KevinBu/Desktop/clemente_lab/CUTIE/tests/lungpt_1pc_point_unit_test_kkc1fdr0.05/test_config.ini






