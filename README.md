# CUTIE #

CUTIE is a python module for evaluating pairwise correlations in a dataset for
influential points using various methods, including CUTIE, jackknifing, and
bootstrapping.

Detailed information about the package can be found in the accompanying publication.

Questions and concerns should be sent to kbu314@gmail.com.


## Requirements ##
---
CUTIE is currently only supported on Linux/Unix-based systems.

## Dependencies ##
---
CUTIE is written for python 3.7 and requires the following packages
1. click
2. numpy
3. pandas
4. statsmodels
5. scipy
6. matplotlib
7. seaborn
8. py
9. pytest

## Setup ##
---
1. Clone this repository `https://github.com/clemente-lab/CUTIE.git` into desired <install_path>
2. Install Anaconda3: `https://www.anaconda.com/distribution/`
3. Create a conda environment for CUTIE: `conda create -n 'cutie' python=3.7 click numpy pandas statsmodels scipy matplotlib seaborn py pytest`
4. Activate the conda environment: `conda activate cutie`
5. Install dependencies: `conda install -c anaconda click numpy pandas statsmodels scipy matplotlib seaborn py pytest`
6. In CUTIE's install directory, run: `python3 setup.py install --prefix=<your_install_directory`. You may need to set your `$PYTHONPATH` variable accordingly, e.g. `export PYTHONPATH=/Users/<user>/tools/sandbox/lib/python3.7/site-packages/` then `Python setup.py install  --prefix=/Users/<user>/tools/sandbox/`

### Usage ###
---

The config file test_config.ini will need to be modified depending on the location of the input data files and the desired result for the output directory. In any given config file we have the given fields (see `/examples/config_template.ini`). Note that tidy dataframes have samples as rows, variables as columns; untidy denotes the converse. When you are ready, simply run `calculate_cutie.py -i <path_to_config_file>/<config_file>`.

```
[input]
samp_var1_fp: <path_to_df/df.csv>
delimiter1: <delimiter type e.g. ,>
samp_var2_fp: <path_to_df/df.txt>
delimiter2: <delimiter type e.g. \t>
f1type: <tidy or untidy>
f2type: <tidy or untidy>
skip1: <integer>
skip2: <integer>
startcol1: <integer>
endcol1: <integer>
startcol2: <integer>
endcol2: <integer>
paired: <True or False>
overwrite: <True or False>

[output]
working_dir: <your_path/>

[stats]
param: <p or r>
statistic: <pearson, rpearson, spearman, rspearman, kendall, rkendall>
resample_k: <integer, default 1>
alpha: <float, default 0.05>
mc: <nomc, fdr, fwer, or bonferroni>
fold: <True or False>
fold_value: <integer, default 1>
corr_compare: <True or False>

[graph]
graph_bound: <integer, default 30>
fix_axis: <True or False>
```




## Example usage ##
---
1. Modify the config file `<install_path>/CUTIE/examples/test_p_config_local.txt` so that fields `samp_var1_fp` matches `<install_path>/CUTIE/examples/CUTIE/tests/otu_table_MultiO_merged___L6.txt
 `samp_var2_fp` matches `<install_path>/CUTIE/tests/Mapping.Pneumotype.Multiomics.RL.NYU.w_metabolites.w_inflamm.txt>` and `working_dir` matches `<install_path>/CUTIE/tests/lungpt_1pc_p_point_unit_test0.05/`.
2. Run `calculate_cutie.py -i <install_path>/CUTIE/tests/test_p_config_local.txt`





