# CUTIE #

CUTIE is a python module for evaluating pairwise correlations in a dataset for influential points via resampling. Detailed information about the package can be found in the accompanying publication.

Questions and concerns should be sent to kbu314@gmail.com.

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
6. In CUTIE's install directory, run: `python3 setup.py install`

### Usage ###
---

The config file test_config.ini will need to be modified depending on the location of the input data files and the desired result for the output directory. In any given config file we have the given fields (see `/demos/config_template.ini`). Note that tidy dataframes have samples as rows, variables as columns; untidy denotes the converse. When you are ready, simply run `calculate_cutie.py -i <path_to_config_file>/<config_file>`. See the Wiki for more information and detailed examples.

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






