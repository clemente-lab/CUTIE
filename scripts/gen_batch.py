import click
import os
import glob
from collections import defaultdict

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='0.1')

# Required arguments
@click.option('-s', '--max_seed', type=int,
              help='max int seed number')
@click.option('-w', '--working_dir', type=click.Path(exists=True),
              help='working dir to put batch jobs')
@click.option('-i', '--input_dir', type=click.Path(exists=True),
              help='input dir with configs')

def gen_batch(max_seed, working_dir, input_dir):
    dirs = glob.glob(input_dir + '*')

    if not os.path.exists(working_dir + 'batch_jobs/'):
        os.makedirs(working_dir + 'batch_jobs/')

    seed_to_dirs = defaultdict(list)

    for d in dirs:
        # fp = p_nomc_1_rpearson_False_4_NP_25_0.9
        fp = os.path.basename(d)
        seed = fp.split('_')[5]
        with open(d + '/commands_' + fp + '.txt', 'r') as f:
            line = f.readline()
            command = line.split('&&')[-1]
            seed_to_dirs[seed].append(command)

    for s in range(max_seed):
        with open(working_dir + 'batch_jobs/batch_' + str(s) + '.txt', 'w') as f:
            f.write('export PYTHONPATH=$PYTHONPATH:/hpc/users/buk02/tools/sandbox/lib/python3.7/site-packages/')
            for c in seed_to_dirs[str(s)]:
                f.write(' && ')
                f.write(c)

if __name__ == "__main__":
    gen_batch()
