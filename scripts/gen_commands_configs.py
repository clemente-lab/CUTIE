import glob
import os
import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='0.1')

# Required arguments
@click.option('-fv', '--fold_value', type=str,
              help='fold value for criterion for p value change')
@click.option('-p', '--param', type=str,
              help='string denoting type of param used')
@click.option('-s', '--statistic', type=str,
              help='string denoting type of analysis')
@click.option('-m', '--multi_corr', type=str,
              help='string denoting type of multiple corrections')
@click.option('-c', '--corr_compare', type=str,
              help='boolean denoting whether performing cooksd or not')
@click.option('-w', '--working_dir', type=click.Path(exists=True),
              help='working dir to save results')
@click.option('-i', '--input_dir', type=click.Path(exists=True),
              help='input dir with .txt files of data')
@click.option('-o', '--output_dir', type=click.Path(exists=True),
              help='output dir to put config files')

def gen_commands_configs(param, fold_value, statistic, multi_corr, corr_compare,
                         working_dir, input_dir, output_dir):
    fv = fold_value
    files = glob.glob(input_dir + '*.txt')
    for fp in files:
        fn = os.path.basename(fp)
        if statistic != 'pearson':
            corr_compare = 'False'
        f_id = param + '_' + multi_corr + '_' + fv + '_' + statistic + '_' + corr_compare + '_' + os.path.splitext(fn)[0]
        out_dir = output_dir + f_id + '/'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        working_outdir = working_dir + f_id + '/'
        if not os.path.isdir(working_outdir):
            os.mkdir(working_outdir)
        with open(out_dir + 'config_' + f_id + '.txt','w') as f:
            f.write('[input]')
            f.write('\n')
            f.write('samp_var1_fp: ' + fp)
            f.write('\n')
            f.write('delimiter1: \\t')
            f.write('\n')
            f.write('samp_var2_fp: ' + fp)
            f.write('\n')
            f.write('delimiter2: \\t')
            f.write('\n')
            f.write('f1type: tidy')
            f.write('\n')
            f.write('f2type: tidy')
            f.write('\n')
            f.write('skip1: 0')
            f.write('\n')
            f.write('skip2: 0')
            f.write('\n')
            f.write('startcol1: -1')
            f.write('\n')
            f.write('endcol1: -1')
            f.write('\n')
            f.write('startcol2: -1')
            f.write('\n')
            f.write('endcol2: -1')
            f.write('\n')
            f.write('paired: True')
            f.write('\n')
            f.write('overwrite: True')
            f.write('\n')
            f.write('\n')
            f.write('[output]')
            f.write('\n')
            f.write('working_dir: ' + working_outdir)
            f.write('\n')
            f.write('\n')
            f.write('[stats]')
            f.write('\n')
            f.write('param: ' + param)
            f.write('\n')
            f.write('statistic: ' + statistic)
            f.write('\n')
            f.write('resample_k: 1')
            if param == 'p':
                f.write('\n')
                f.write('alpha: 0.05')
            elif param == 'r':
                f.write('\n')
                f.write('alpha: 0.50')
            f.write('\n')
            f.write('mc: ' + multi_corr)
            f.write('\n')
            f.write('fold: True')
            f.write('\n')
            f.write('fold_value: ' + fv)
            f.write('\n')
            f.write('corr_compare: ' + corr_compare)
            f.write('\n')
            f.write('\n')
            f.write('[graph]')
            f.write('\n')
            f.write('graph_bound: 30')
            f.write('\n')
            f.write('fix_axis: False')

        with open(out_dir + 'commands_' + f_id + '.txt','w') as f:
            f.write('export PYTHONPATH=$PYTHONPATH:/hpc/users/buk02/tools/sandbox/lib/python3.7/site-packages/ && python /sc/hydra/work/buk02/CUTIE/scripts/calculate_cutie.py -i ' + out_dir + 'config_' + f_id + '.txt')

if __name__ == "__main__":
    gen_commands_configs()
