###
# Installation, setup, and preprocessing
###

python setup.py install <install-directory>

# minerva files
/sc/orga/work/buk02/clemente_lab/lungpt_data/otu_table_MultiO_merged___L6.txt
/sc/orga/work/buk02/clemente_lab/lungpt_data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt
/sc/orga/work/buk02/clemente_lab/cutie/data/pre_sparcc_MSQ/otu_table.MSQ34_L6.txt

# minepy installation
http://minepy.sourceforge.net/docs/0.3.5/install.html

# install
cd /Users/KevinBu/Desktop/clemente_lab/CUTIE
export PYTHONPATH=/Users/KevinBu/tools/sandbox/lib/python3.7/site-packages/
Python3 setup.py install  --prefix=/Users/KevinBu/tools/sandbox/

# navigate to dir
cd /Users/KevinBu/Desktop/clemente_lab/CUTIE/tests/

# mk test directory
mkdir /Users/KevinBu/Desktop/clemente_lab/CUTIE/tests/lungpt_1pc_point_unit_test0.05/

# run cutie
python3 /Users/KevinBu/Desktop/clemente_lab/CUTIE/scripts/calculate_cutie.py -df /Users/KevinBu/Desktop/clemente_lab/CUTIE/tests/config_defaults.ini -cf /Users/KevinBu/Desktop/clemente_lab/CUTIE/tests/lungpt_1pc_point_unit_test_kkc1fdr0.05/test_config.ini






