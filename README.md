python setup.py install

merge_otu_tables.py -i otu_table_MultiO__Status_Smoker_Non.Smoker___L6.biom,otu_table_MultiO__Status_Smoker_Smoker___L6.biom -o otu_table_MultiO_merged___L6.biom

biom convert -i otu_table_MultiO_merged___L6.biom -o otu_table_MultiO_merged___L6.txt --to-tsv

python scripts/stats_cutie.py -sm data/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt -sb data/otu_table_MultiO_merged___L6.txt
