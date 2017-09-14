###
# Installation, setup, and preprocessing
###
python setup.py install

git rm --cached -r .
git add cutie.ipynb
git add scripts/.
git add cutie/.
git add setup.py


###
# CUtIe analysis of Lung Pneumotype Data (L6 only, L7 not attempted yet)
###

# MERGE AND CONVERT FILES 
merge_otu_tables.py -i otu_table_MultiO__Status_Smoker_Non.Smoker___L6.biom,otu_table_MultiO__Status_Smoker_Smoker___L6.biom -o otu_table_MultiO_merged___L6.biom

biom convert -i otu_table_MultiO_merged___L6.biom -o otu_table_MultiO_merged___L6.txt --to-tsv


###
# CUtIe resample k = 1 comparison to other pointwise metrics
### 

mkdir data_analysis/lungptL6_pointcomparison_pc1fdr0.05
python scripts/lungbactmeta_cutiepc.py -sm data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt -sb data/otu_table_MultiO_merged___L6.txt -d data_analysis/lungptL6_pointcomparison_pc1fdr0.05/ -a 0.05 --fdr


###
# CUtIe ANALYSIS 
###

# lung bact meta with resample k = 1
mkdir data_analysis/lungptL6_kpc1fdr0.05
python scripts/lungbactmeta_cutiekpc.py -sm data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt -sb data/otu_table_MultiO_merged___L6.txt -d data_analysis/lungptL6_kpc1fdr0.05/ -k 1 -a 0.05 --fdr

# lung bact meta with resample k = 3
mkdir data_analysis/lungptL6_kpc3fdr0.05
python scripts/lungbactmeta_cutiekpc.py -sm data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt -sb data/otu_table_MultiO_merged___L6.txt -d data_analysis/lungptL6_kpc3fdr0.05/ -k 3 -a 0.05 --fdr


###
# Proportionality, Concordance and CUtIe PC and SC
###

# proportionality testing
mkdir data_analysis/lungbact_prop0.05
python scripts/lungbact_cutiekprop.py -b data/otu_table_MultiO_merged___L6.txt -d data_analysis/lungbact_prop0.05/ -k 1

# concordance testing
mkdir data_analysis/lungbact_conc0.90
python scripts/lungbact_cutiekconc.py -b data/otu_table_MultiO_merged___L6.txt -d data_analysis/lungbact_conc0.90/ -k 1

# CUtIe PC
mkdir data_analysis/lungbactcorr_cutiekpc1fdr0.05
python scripts/lungbactcorr_cutiekpc.py -b data/otu_table.MSQ34_L6.txt -d data_analysis/lungbactcorr_cutiekpc1fdr0.05/ -k 1 -a 0.05 --fdr

# CUtIe SC
mkdir data_analysis/lungbactcorr_cutieksc1fdr0.05
python scripts/lungbactcorr_cutieksc.py -b data/otu_table.MSQ34_L6.txt -d data_analysis/lungbactcorr_cutieksc1fdr0.05/ -k 1 -a 0.05 --fdr






###
# SparCC on Minerva
###

module load qiime/1.9.1 (1.8.0 is default)
make_otu_network.py -i otu_table.biom -m Fasting_Map.txt -o otu_network



#!/bin/bash
#BSUB -q premium
#BSUB -W 24:00
#BSUB -J sparcc1
#BSUB -P acc_clemej05a
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -m manda
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash
module load sparcc
module load py_packages/2.7-gpu
SparCC.py /sc/orga/work/buk02/clemente_lab/otu_table_MultiO_merged___L6.txt -i 5 --cor_file=/sc/orga/work/buk02/clemente_lab/pre.cor_sparcc1.out

Bsub < sparcc1.lsf

# examples
python SparCC.py example/fake_data.txt -i 5 --cor_file=example/basis_corr/cor_sparcc.out
python SparCC.py example/fake_data.txt -i 5 --cor_file=example/basis_corr/cor_pearson.out -a pearson
python SparCC.py example/fake_data.txt -i 5 --cor_file=example/basis_corr/cor_spearman.out -a spearman

python MakeBootstraps.py example/fake_data.txt -n 5 -t permutation_#.txt -p example/pvals/

python SparCC.py example/pvals/permutation_0.txt -i 5 --cor_file=example/pvals/perm_cor_0.txt
python SparCC.py example/pvals/permutation_1.txt -i 5 --cor_file=example/pvals/perm_cor_1.txt
python SparCC.py example/pvals/permutation_2.txt -i 5 --cor_file=example/pvals/perm_cor_2.txt
python SparCC.py example/pvals/permutation_3.txt -i 5 --cor_file=example/pvals/perm_cor_3.txt
python SparCC.py example/pvals/permutation_4.txt -i 5 --cor_file=example/pvals/perm_cor_4.txt

python PseudoPvals.py example/basis_corr/cor_sparcc.out example/pvals/perm_cor_#.txt 5 -o example/pvals/pvals.one_sided.txt -t one_sided
python PseudoPvals.py example/basis_corr/cor_sparcc.out example/pvals/perm_cor_#.txt 5 -o example/pvals/pvals.one_sided.txt -t two_sided

###
# Bact phage 
###

mv 1-s2.0-S0092867416303993-mmc3.txt 1-s2.0-S0092867416303993-mmc3_cell.txt

mkdir data_analysis/bactphage_kpc1fdr0.05
python scripts/bactphage_cutiekpc.py -b data/1-s2.0-S0092867416303993-mmc3_cell.txt -d data_analysis/bactphage_kpc1fdr0.05/ -k 1 -a 0.05 --fdr

mkdir data_analysis/bactphage_ksc1fdr0.05
python scripts/bactphage_cutieksc.py -b data/1-s2.0-S0092867416303993-mmc3_cell.txt -d data_analysis/bactphage_ksc1fdr0.05/ -k 1 -a 0.05 --fdr


###
# Bactcorr (old SparCC) comparison of sc and pc
###

mkdir data_analysis/bactcorr_cutieksc1fdr0.05
python scripts/bactcorr_cutieksc.py -b data/otu_table.MSQ34_L6.txt -d data_analysis/bactcorr_cutieksc3fdr0.05/ -k 1 -a 0.05 --fdr

mkdir data_analysis/bactcorr_cutiekpc1fdr0.05
python scripts/bactcorr_cutiekpc.py -b data/otu_table.MSQ34_L6.txt -d data_analysis/bactcorr_cutiekpc1fdr0.05/ -k 1 -a 0.05 --fdr


###
# SPEED TESTING
###
mkdir testdata
python scripts/test_SLR.py -sm testdata/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflammtest.txt -sb testdata/otu_table_MultiO_merged___L6a.txt -nb 200 -n 5 -sc 17 -ec 27 -d testdata/









