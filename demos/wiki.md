**Correlations Under The InfluencE (CUTIE)**
============================================

Welcome ot he CUTIE tutorial and wiki, which will provide software,
docuemntation, and tutorials for this statistical resampling method developed by
the [Clemente Lab](). CUTIE is implemented in Python.

**If you use CUTIE in your work, please cite the paper:**

**CUTIE**
=========

CUTIE computes all pairwise correlations in a given dataset and determines which
of those initially significant correlations are potentially driven by outliers,
and can additionally rescue correlations deemed initially significant due to
these influential points.

CUTIE is available as a Python package. Any questions or concerns can be sent to
kbu314@gmail.com.

------------------------------------------------------------------------

Table of contents
- [**Overview**](#overview)
- [**1. Installation**](#1-lefse--galaxy-)
- [**2. Tutorial**](#2-lefse--conda-docker-vm-)
- [**3. Interpreting Results**](#3-visualization)
- [**Notes**](#notes)

------------------------------------------------------------------------

## **Overview**

The following figure shows CUTIE's workflow.

[![F1a.pdf]('https://github.com/clemente-lab/CUTIE/demos/images/F1b.pdf)]('https://github.com/clemente-lab/CUTIE/demos/images/F1b.pdf)

------------------------------------------------------------------------

## **1. Installation **

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
1. Clone this repository `https://github.com/clemente-lab/CUTIE.git` into a desired <install_path>
2. Install Anaconda3: `https://www.anaconda.com/distribution/`
3. Create a conda environment for CUTIE: `conda create -n 'cutie' python=3.7 click numpy pandas statsmodels scipy matplotlib seaborn py pytest`
4. Activate the conda environment: `conda activate cutie`
5. Install dependencies: `conda install -c anaconda click numpy pandas statsmodels scipy matplotlib seaborn py pytest`
6. In CUTIE's install directory, run: `python3 setup.py install --prefix=<your_install_directory`. You may need to set your `$PYTHONPATH` variable accordingly, e.g. `export PYTHONPATH=/Users/<user>/tools/sandbox/lib/python3.7/site-packages/` then `Python setup.py install  --prefix=/Users/<user>/tools/sandbox/`

### Usage ###
---

The config file test_config.ini will need to be modified depending on the location of the input data files and the desired result for the output directory. In any given config file we have the given fields (see `CUTIE/demos/config_template.ini`). Note that tidy dataframes have samples as rows, variables as columns; untidy denotes the converse. When you are ready, simply run `calculate_cutie.py -i <path_to_config_file>`.

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

------------------------------------------------------------------------

## **2. Tutorial**

In this tutorial, we will be using an example dataset from [lung pneumotyping paper]
and use CUTIE to classify bacteria-metabolite correlations as TP or FP. The bacterial
data is stored as a tsv OTU-table in conventional untidy format (samples as columns,
taxanomic identifiers as rows) while the metabolite data is stored in tidy format
(samples as rows, variables as columns).



## Example usage ##
---
1. Modify the config file `CUTIE/demos/tutorial_config.ini` so that the fields under [input] match `samp_var1_fp: <install_path>/CUTIE/demos/data/otu_table.tsv` and `samp_var2_fp: <install_path>/CUTIE/demos/metabolite_table.tsv`.
Additionally, modify the working directory to `<install_path>/CUTIE/demos/otu_metabolite_tutorial/`.
2. Activate the conda environment `conda activate cutie` from the setup above.
3. Run `calculate_cutie.py -i <install_path>/CUTIE/demos/tutorial_config.txt.` This should take about a minute.


## **3. Interpreting Results**

CUTIE's output includes a variety of diagnostic features aimed at elucidating sources of outllier bias behind
pairwise correlations in a given dataset. From the tutorial above, you should see the following:

- A log file, <log.txt>, which indicates what CUTIE has parsed; information is included such as # of variables, samples, and
their string identifiers.
- A directory `data_processing` with 4 txt files: (1) `counter_samp_number_resample1.txt`, indicating the number of CUTIE's
(FP) to which each sample contributes, (2, 3) `counter_var1_number_resample1.txt` and `counter_var1_number_resample2.txt`, the analogous files for dataframe1 (otu) and dataframe2 (metabolite), and finally (4) `summary_df_resample_1.txt`, which
contains more detailed information regarding each pairwise correlation. Examining the file for this example, we have
the following headers:

```var1_index  var2_index  pvalues correlations    r2vals  indicators  TP_rev_indicators   FP_rev_indicators   extreme_p   extreme_r   p_ratio r2_ratio    ['cutie_1pc']   ['cookd']   ['dffits']  ['dsr'] ['cutie_1pc', 'cookd']  ['cutie_1pc', 'dffits'] ['cutie_1pc', 'dsr']    ['cookd', 'dffits'] ['cookd', 'dsr']    ['dffits', 'dsr']   ['cutie_1pc', 'cookd', 'dffits']    ['cutie_1pc', 'cookd', 'dsr']   ['cutie_1pc', 'dffits', 'dsr']  ['cookd', 'dffits', 'dsr']  ['cutie_1pc', 'cookd', 'dffits', 'dsr']```

- Each row in the dataframe represents a pairwise correlation.
- `var1_index` and `var2_index` denote the indicies (0-indexed) of the variables for a given pairwise correlation.
- `pvalues`, `correlations`, and `r2vals` denote the original p-value, r-value, and r-squared for that correlation
- indiciators is 0 if that correlation was never assessed (p > 0.05), 1 if it is a TP, and -1 if it is an FP according to CUTIE
- `TP_rev_indicators` and `FP_rev_indicators` are 1 if the correlation was a reverse-sign correlation of the TP or FP class respectively, 0 if not
- `extreme_p` and `extreme_r` denote the most extreme p and r-value obtained respectively when omitting a point (highest p-value if performing TP/FP classification, lowest p-value obtained if performing TN/FN classification)
- `p_ratio` and `r2_ratio` represent the ratio between the most extreme p and r-squared value and the original value.
- The bracket/list values represent areas on a venn-diagram and only appear because corr_compare was set to True in the original config. Here, a value can be 0, 1 or -1 for a given correlation; 0 denotes that correlation was never assessed (never significant), -1 if that correlation is a FP as considered by the metrics in that region (e.g. a -1 under `['cookd','dffits','dsr']` would indicate that correlation was a false positive to those three metrics, but not CUTIE) and 1 otherwise.

- graphs contains a variety of exploratory visualizations

`true_corr_rev_<class>_<k>_<n>_revsign` such as `true_corr_rev_TP_1_0_revsign` and similarly named folders contain randomly sampled scatterplots from correlations classified as True POsitives and exhibit a sign-reversal; k indicates # of points being resampled (here k = 1) and n denotes the number of scatterplots in this class (here n = 0).

folders such as `<metrics>_<class>_<k>_<n>` such as `['cutie_1pc', 'dsr']_TP_1_165` and `['cutie_1pc', 'dsr']_FP_1_0` indicate the number of correlations belonging to that class according to such metrics (e.g. 165 correlations were flagged as TP by CUTIE and DSR but not by any other metric.

plots such as `sample_corr_<metrics>_<class>_<k>.pdf` e.g. `sample_corr_['cutie_1pc', 'cookd', 'dffits', 'dsr']_FP_1.pdf` indicate the distribution of sample correlation coefficients in set of FP correlations as flagged by all four metrics combined.

plots such as `<class>_<k>_pvalues.pdf`, `<class>_<k>_log_fold_values`, or `<class>_<k>_log_pvalues` indicate the distribution of those parameters in a given class with k points resampled.

the set of three plots, `counter_samp_number_resample1.pdf`, `counter_var1_number_resample1.pdf` and `counter_var2_number_resample1.pdf` all plot the distribution from the datafarmes in data_processing, i.e. the number of CUTIEs (FPs) to which each sample or variable contributes.


[![hmp\_aerobiosis\_small.cladogram.png](https://github.com/biobakery/biobakery/blob/master/images/2429145998-hmp_aerobiosis_small.cladogram.png)](https://github.com/biobakery/biobakery/blob/master/images/2429145998-hmp_aerobiosis_small.cladogram.png)

------------------------------------------------------------------------

Notes
-----

-   In case of any issue with any tool, please feel free post it
    in [bioBakery Support Forum](http://forum.biobakery.org/).
-   [Source code repository](https://github.com/biobakery)
