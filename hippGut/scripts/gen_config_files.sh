#!/bin/bash

work_path=$(pwd)/..

input_dir=$work_path/inputs/var_files
output_dir=$work_path/inputs/config_files
mkdir -p $output_dir
rm $output_dir/*

groups="Oral_Cases Oral_Controls Stool_Cases Stool_Controls"
levels="L5 L6"

for group in $groups; do
    for level in $levels; do
        fp_1=$input_dir/${group}_${level}_otu.txt
        fp_2=$input_dir/${group}_${level}_cytokine.txt
        
        config_path=$output_dir/config_${group}_${level}.ini

        cat <<EOF > $config_path
[input]
samp_var1_fp: $fp_1
delimiter1: \t
samp_var2_fp: $fp_2
delimiter2: \t
f1type: tidy
f2type: tidy
skip1: 0
skip2: 0
startcol1: -1
endcol1: -1
startcol2: -1
endcol2: -1
paired: False

[output]
working_dir: $work_path/outputs/${group}_${level}
overwrite: False

[stats]
param: p
statistic: kendall
resample_k: 1
alpha: 0.05
mc: fdr
fold: False
fold_value: 1
corr_compare: False

[graph]
graph_bound: 30
fix_axis: False

EOF
    done
done
