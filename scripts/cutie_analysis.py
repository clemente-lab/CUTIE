from glob import glob
from subprocess import call

paths = glob('*')

# paths = ['cutie_hdac_rpc1fdr0.05', 'sim_copula_n50_norm_0_1_ksc1fdr0.05']
for path in paths:
    # make directory in work 
    try:
        call(['mkdir', '/sc/orga/work/buk02/data_analysis/' + path])
    except:
        pass

    # make commands file
    with open('/sc/orga/work/buk02/data_analysis/' + path + '/commands.txt', 'w') as f:
        f.write('export PYTHONPATH=$PYTHONPATH:/hpc/users/buk02/tools/sandbox/lib/python2.7/site-packages/ ' + \
            '&& python -W ignore /sc/orga/work/buk02/cutie/scripts/calculate_cutie.py ' + \
            '-df /sc/orga/work/buk02/cutie/scripts/config_defaults.ini -cf ' + \
            '/sc/orga/work/buk02/data_analysis/' + path + '/config.ini')

    # make directory in scratch
    try:
        call(['mkdir', '/sc/orga/scratch/buk02/data_analysis/' + path])
    except:
        pass
    
    # generate lsf file
    # call(['module load python/2.7.14'])
    # call(['module load py_packages/2.7'])

    # submit job
    call(['python /sc/orga/projects/clemej05a/labtools/scripts/generate_lsf.py -c ' + \
        '/sc/orga/work/buk02/data_analysis/' + path + '/commands.txt -m qiime/1.9.1 ' + \
         '-N cutie -o /sc/orga/work/buk02/data_analysis/' + path + '/ -n 2 -w 48:00 ' + \
         '-s True'], shell = True)

