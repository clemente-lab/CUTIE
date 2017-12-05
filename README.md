###
# Installation, setup, and preprocessing
###

python setup.py install <install-directory>


###############
# AlCoB paper #
###############

###
# Lung pneumotyping with Pearson
###

# MERGE AND CONVERT FILES 
merge_otu_tables.py -i otu_table_MultiO__Status_Smoker_Non.Smoker___L6.biom,otu_table_MultiO__Status_Smoker_Smoker___L6.biom -o otu_table_MultiO_merged___L6.biom

biom convert -i otu_table_MultiO_merged___L6.biom -o otu_table_MultiO_merged___L6.txt --to-tsv

# DATA ANALYSIS DIRECTORY
mkdir data_analysis/alcob_lung_pointwise_kpc1fdr0.05
python -W ignore scripts/calculate_cutie.py -l L6 -s1 data/lung_pt/otu_table_MultiO_merged___L6.txt -s2 data/lung_pt/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt --otu1 --map2 -d data_analysis/alcob_lung_pointwise_kpc1fdr0.05/ -sc 17 -ec 100 -stat kpc --corr -k 1 -p False -a 0.05 --fdr

###
# Lung pneumotyping with Spearman (-nomc, as -fdr yields no significant results)
### 

# DATA ANALYSIS DIRECTORY
mkdir data_analysis/alcob_MSQ_pointwise_kpc1fdr0.05
python -W ignore scripts/calculate_cutie.py -l MSQ -s1 data/pre_sparcc_MSQ/otu_table.MSQ34_L6.txt -s2 data/pre_sparcc_MSQ/otu_table.MSQ34_L6.txt --otu1 --otu2 -d data_analysis/alcob_MSQ_pointwise_kpc1fdr0.05/ -skip 1 -stat kpc --corr -k 1 -p True -a 0.05 --fdr