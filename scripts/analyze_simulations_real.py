import pandas as pd
import numpy as np
import glob
np.random.seed(0)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import click


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='0.1')

# Required arguments
@click.option('-fv', '--fold_value', type=str,
              help='fold value for criterion for p value change')
@click.option('-s', '--statistic', type=str,
              help='string denoting type of analysis')
@click.option('-m', '--multi_corr', type=str,
              help='string denoting type of multiple corrections')
@click.option('-p', '--param', type=str,
              help='string denoting params used')
@click.option('-c', '--corr_compare', type=str,
              help='boolean denoting whether performing cooksd or not')
@click.option('-i', '--input_dir', type=click.Path(exists=True),
              help='input dir with .txt files of data')
@click.option('-o', '--output_dir', type=click.Path(exists=True),
              help='output dir to put config files')


def analyze_simulations_real(fold_value, statistic, multi_corr, param,
                             corr_compare, input_dir, output_dir):
    '''
    Script for analysis of real data by CUTIE
    '''

    def parse_log(f, cookd):
        lines = [l.strip() for l in f.readlines()]
        defaulted = False
        if cookd == 'True':
            for l in lines:
                if "defaulted" in l:
                    defaulted = True
                elif "initial_corr" in l:
                    initial_corr = float(l.split(' ')[-1])
                elif "false correlations according to cookd" in l:
                    false_corr = float(l.split(' ')[-1])
                elif "true correlations according to cookd" in l:
                    true_corr = float(l.split(' ')[-1])
                elif "runtime" in l:
                    runtime = float(l.split(' ')[-1])
            rs_false = np.nan
            rs_true = np.nan

        else:
            # check if FDR correction defaulted
            for l in lines:
                if "defaulted" in l:
                    defaulted = True
                elif "initial_corr" in l:
                    initial_corr = float(l.split(' ')[-1])
                elif "false correlations" in l:
                    false_corr = float(l.split(' ')[-1])
                elif "true correlations" in l:
                    true_corr = float(l.split(' ')[-1])
                elif "FP/TN1" in l:
                    rs_false = float(l.split(' ')[-1])
                elif "TP/FN1" in l:
                    rs_true = float(l.split(' ')[-1])
                elif "runtime" in l:
                    runtime = float(l.split(' ')[-1])

        return defaulted, initial_corr, false_corr, true_corr, rs_false, rs_true, runtime


    headers = [
        'analysis_id',
        'parameter',
        'distribution',
        'statistic',
        'mc_used', #NEW
        'fold_value', # NEW
        'pointwise', #NEW
        'defaulted', # binary
        'initial_corr',
        'true_corr(TP_FN)',
        'false_corr(FP_TN)',
        'rs_true_corr_TP_FN',
        'rs_false_corr_FP_TN',
        'runtime'
    ]

    # populate df
    results_df = pd.DataFrame()

    mcs = multi_corr.split(',')
    fvs = fold_value.split(',')
    stats = statistic.split(',')
    cds = corr_compare.split(',')
    ds = ['lungtx', 'lungpt', 'lungc','who','hdac']
    for p in param:
        for mc in mcs:
            for fv in fvs:
                for s in stats:
                    for cd in cds:
                        for d in ds:
                            # nomc_10_pearson_True_lungpt
                            analysis_id = '_'.join([p, mc, fv, s, cd, d])
                            path = input_dir + analysis_id + '/'
                            files = sorted(glob.glob(path + '*.txt'))
                            # grab most recent log file
                            try:
                                rel_logfile = files[-1]
                                with open(rel_logfile, 'r') as f:
                                    defaulted, initial_corr, false_corr, \
                                        true_corr, rs_false, rs_true, runtime = parse_log(f,cd)

                                    new_row = pd.DataFrame([[analysis_id, p, d, s,
                                                            mc, fv, cd, defaulted,
                                                            initial_corr, true_corr,
                                                            false_corr, rs_true,
                                                            rs_false, runtime]],
                                                            columns=headers)

                                    results_df = results_df.append(new_row)
                            except:
                                print(analysis_id)
                                print('Failed parsing')
                                if cd == 'True':
                                    if s == 'pearson':
                                        print(analysis_id)
                                else:
                                    print(analysis_id)

    colnames = ['LungTranscriptomics', 'Micrometa', 'Microbiome', 'Gene Expression', 'WHO']

    col_to_corr = {
        'LungTranscriptomics': 292 * 97, #depends on sum vs unsum
        'Micrometa': 83 * 897,
        'Microbiome': 748 * 747 / 2,
        'Gene Expression': 1000 * 999 / 2,
        'WHO': 354 * 353 / 2
    }

    dists = ['lungtx', 'lungpt', 'lungc', 'hdac', 'who']

    dist_to_corr = {
        'lungtx': 292 * 97,
        'lungpt': 83 * 897,
        'lungc': 748 * 747 / 2,
        'hdac': 1000 * 999 / 2,
        'who': 354 * 353 / 2
    }
    results_df.to_csv(output_dir + 'real_df.txt', sep='\t')

    # populate indices and ids for the dfs
    for p in param:
        for fv in fvs:
            for mc in mcs:
                indices = []
                ids = []
                indices.append('_'.join(['pearson', 'cd', fv, mc, p]))
                indices.append('Pct initial corr')
                ids.append('_'.join([mc, fv, 'pearson', 'True', p]))
                for stat in stats:
                    indices.append('_'.join([stat, fv, mc]))
                    indices.append('Pct initial corr')
                    ids.append('_'.join([mc, fv, stat, 'False']))

                # populate  df
                df_array = []
                for i, (idstring, index) in enumerate(zip(ids, indices)):
                    row_fracs = []
                    mc, fv, s, cd = idstring.split('_')
                    for dist in dists:
                        row = results_df[(results_df['parameter'] == p) & (results_df['distribution'] == dist) & (results_df['statistic'] == s) \
                                     & (results_df['mc_used'] == mc) & (results_df['fold_value'] == fv) & (results_df['pointwise'] == cd)]
                        try:
                            row_fracs.append(float(row['true_corr(TP_FN)'] /row['initial_corr'].values)) # correctly id tp
                        except:
                            row_fracs.append(np.nan)
                            print('nan in row fracs')
                            print(dist, idstring)

                    df_array.append(row_fracs)

                    initial_sig_fracs = []
                    for dist in dists:
                        row = results_df[(results_df['distribution'] == dist) & (results_df['statistic'] == s) \
                                     & (results_df['mc_used'] == mc) & (results_df['fold_value'] == fv) & (results_df['pointwise'] == cd)]
                        # change number 249500 to n_corr depending on dataset
                        try:
                            initial_sig_fracs.append(float(row['initial_corr'] / dist_to_corr[dist]))
                        except:
                            initial_sig_fracs.append(np.nan)

                    df_array.append(initial_sig_fracs)

                pie_df = pd.DataFrame(data = df_array, index = indices, columns = colnames)
                pie_df = pie_df.rename_axis('Statistic')
                pie_df = pie_df.apply(pd.to_numeric).round(2)

                # parse the reverse sign shenanigans
                df_array = []

                # cut out the cookd parts
                rs_ids = ids[-len(stats):]
                rs_indices = indices[-2*len(stats):]
                for i, (idstring, index) in enumerate(zip(rs_ids, rs_indices)):
                    # stat = 'Pearson'
                    row_fracs = []
                    mc, fv, s, cd, p = idstring.split('_')
                    for dist in dists:
                        row = results_df[(results_df['parameter'] == p) & (results_df['distribution'] == dist) & (results_df['statistic'] == s) \
                                     & (results_df['mc_used'] == mc) & (results_df['fold_value'] == fv) & (results_df['pointwise'] == 'False')]
                        try:
                            row_fracs.append(float(row['rs_true_corr_TP_FN'] /row['initial_corr'].values)) # correctly id tp
                        except:
                            row_fracs.append(np.nan)
                            print('failed to parse rs')
                            print(dist, idstring)

                    df_array.append(row_fracs)

                    initial_sig_fracs = []
                    for dist in dists:
                        row = results_df[(results_df['parameter'] == p) & (results_df['distribution'] == dist) & (results_df['statistic'] == s) \
                                     & (results_df['mc_used'] == mc) & (results_df['fold_value'] == fv) & (results_df['pointwise'] == 'False')]
                        # change number 249500 to n_corr depending on dataset
                        try:
                            initial_sig_fracs.append(float(row['initial_corr'] / dist_to_corr[dist]))
                        except:
                            initial_sig_fracs.append(np.nan)

                    df_array.append(initial_sig_fracs)

                rs_df = pd.DataFrame(data = df_array, index = rs_indices, columns = colnames)
                rs_df = rs_df.rename_axis('Statistic')
                rs_df = rs_df.apply(pd.to_numeric).round(2)

                # currently the four dfs are
                # pie_df and rs_df
                # only pie_df has cookd info in it
                # the outer loop has mc and fv so when you save fig make sure to incl those

                # dictionary from which to get results for pie plots
                dd = {}

                # cut out micrometa dataset
                pie_df = pie_df.drop(['Micrometa'],axis=1)
                nocd_pie_df = pie_df.iloc[2:,:]
                rs_df = rs_df.drop(['Micrometa'],axis=1)
                sub_colnames = ['LungTranscriptomics', 'Microbiome', 'Gene Expression', 'WHO']

                # obtain indices without cook's D
                vals = list(nocd_pie_df.index.values)
                # skips by 2 (AKA every other)
                new_vals = vals[0::2]
                for v in new_vals:
                    dd[v] = {}

                for v in new_vals:
                    # v = 'pearson_1_fdr'
                    # check to make sure forward direction
                    if v.split('_')[0][0] != 'r':
                        dd[v]['rsTP'] = rs_df.loc[v,:].values
                    else:
                        dd[v]['rsFN'] = rs_df.loc[v,:].values


                for v in new_vals:
                    rows = nocd_pie_df.iloc[vals.index(v):vals.index(v)+2,:].values
                    if v.split('_')[0][0] != 'r':
                        dd[v]['TP'] = rows[0]
                        dd[v]['initial_sig'] = rows[1]
                    else:
                        dd[v]['FN'] = rows[0]
                        dd[v]['initial_insig'] = rows[1]


                for_vals = new_vals[::2]
                v_to_cd = {}
                # just get Cook's D
                # should be '_'.join(['pearson', 'cd', fv, mc])
                # cd_val = list(pie_df.index.values)[0::2][0]

                # first two rows are cd
                rows = pie_df.iloc[0:2,:].values
                v_to_cd['TP'] = rows[0]
                v_to_cd['initial_sig'] = rows[1]

                # create figure
                f, axarr = plt.subplots(len(for_vals) + 1,len(sub_colnames))
                print(dd)

                # iterate over dataset
                for d in range(len(sub_colnames)):
                    labels = ['TP', 'FP', 'N']
                    colors = ['#66b3ff','#ff9999','#FFC000']#,'#ffcc99']
                    TP = v_to_cd['TP'][d]
                    P = v_to_cd['initial_sig'][d]
                    sizes = [TP * P, (1-TP)*P,1-P]

                    axs = axarr[0, d]
                    # note colnames = ['Micrometa', 'Microbiome', 'Gene Expression', 'WHO']
                    title = sub_colnames[d] + ', ' + 'Cook\'s D' + '\n' + str(int(col_to_corr[sub_colnames[d]]))
                    axs.set_title(title)
                    patches, texts, autotexts = axs.pie(sizes, colors = colors, labels=None, autopct='%1.1f%%', startangle=0,
                                                       labeldistance = 1, pctdistance = 1.2)
                    #plt.legend(patches, autotexts, loc='center left', bbox_to_anchor=(-0.1, 1.),fontsize=8)
                    fs = 12
                    ts = 12
                    #patches[0].set_fontsize(fs)
                    #patches[1].set_fontsize(fs)
                    #patches[2].set_fontsize(fs)
                    texts[0].set_fontsize(fs)
                    texts[1].set_fontsize(fs)
                    texts[2].set_fontsize(fs)
                    autotexts[0].set_fontsize(ts)
                    autotexts[1].set_fontsize(ts)
                    autotexts[2].set_fontsize(ts)

                    #draw circle
                    centre_circle = plt.Circle((0,0),0.50,fc='white')
                    fig = plt.gcf()
                    fig.set_size_inches(10,10)
                    #fig.gca().add_artist(centre_circle)
                    axs.add_artist(centre_circle)
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    axs.axis('equal')
                    plt.tight_layout()
                    #plt.show()

                    # iterate over statistic
                    for v in range(len(for_vals)):
                        val = for_vals[v]

                        # labels = ['TP', 'rsTP', 'FP', 'FN', 'rsFN', 'TN']
                        labels = ['TP', 'rsTP', 'FP', 'FN', 'TN']
                        # TP is blue FP is red FN is green TN is purple
                        # for rs case
                        # reverse sign but still true FP is non reverse sign
                        colors = ['#66b3ff','#ADD8E6','#ff9999','#99ff99','#8064A2']
                        TP = dd[val]['TP'][d]
                        rsTP = dd[val]['rsTP'][d]
                        P = dd[val]['initial_sig'][d]
                        FN = dd['r' + val]['FN'][d]
                        rsFN = dd['r' + val]['rsFN'][d]
                        N = dd['r' + val]['initial_insig'][d]
                        # sizes = [(TP - rsTP) * P, rsTP * P,(1-TP)*P, (FN - rsFN) * N, rsFN * N, (1-FN)*N]
                        sizes = [(TP - rsTP) * P, rsTP * P,(1-TP)*P, FN * N, (1-FN)*N]
                        print(sub_colnames[d])
                        print(val)
                        print(labels)
                        print(sizes)

                        # plt.subplot(len(new_vals),len(colnames),i)
                        axs = axarr[v + 1, d]
                        # title = colnames[d] + ', ' + val.split('_')[0] + '\n' + str(int(dist_to_corr[colnames[d]]))
                        # axs.set_title(title)

                        # def draw_pie(sizes, colors):
                        patches, texts, autotexts = axs.pie(sizes, colors = colors, labels=None, autopct='%1.1f%%', startangle=0,
                                                           labeldistance = 1, pctdistance = 1.2)
                        fs = 12
                        ts = 12
                        texts[0].set_fontsize(fs)
                        texts[1].set_fontsize(fs)
                        texts[2].set_fontsize(fs)
                        texts[3].set_fontsize(fs)
                        texts[4].set_fontsize(fs)
                        autotexts[0].set_fontsize(ts)
                        autotexts[1].set_fontsize(ts)
                        autotexts[2].set_fontsize(ts)
                        autotexts[3].set_fontsize(ts)
                        autotexts[4].set_fontsize(ts)

                        #draw circle
                        centre_circle = plt.Circle((0,0),0.50,fc='white')
                        fig = plt.gcf()
                        fig.set_size_inches(10,10)
                        #fig.gca().add_artist(centre_circle)
                        axs.add_artist(centre_circle)
                        # Equal aspect ratio ensures that pie is drawn as a circle
                        axs.axis('equal')
                        plt.tight_layout()

                f.savefig(output_dir + 'pieplots_dfreal_combined_' + p + '_' + mc + '_' + fv + '.pdf')
                plt.close(fig)


if __name__ == "__main__":
    analyze_simulations_real()




