#!/bin/bash

work_path=/sc/arion/projects/clemej05a/jakleen/hippGut/pilot/fig_3b/cutie_analysis/CUTIE/hippGut
input_dir=$work_path/inputs/config_files
job_dir=$work_path/jobs
log_dir=$work_path/logs

rm $job_dir/*
rm $log_dir/*

groups="Oral_Cases Oral_Controls Stool_Cases Stool_Controls"
levels="L5 L6"

for group in $groups; do
    for level in $levels; do
        file_path=$input_dir/config_${group}_${level}.ini
        
        output_dir=$work_path/outputs/${group}_${level}
        rm -r $output_dir
        mkdir $output_dir

        job_path=$job_dir/${group}_${level}.lsf

        cat <<EOF > $job_path
#!/bin/bash
#BSUB -q premium
#BSUB -W 96:00
#BSUB -J ${group}_${level}
#BSUB -P acc_clemej05a
#BSUB -n 1 
#BSUB -R "span[hosts=1]"
#BSUB -R rusage[mem=10000]
#BSUB -o ${log_dir}/${group}_${level}_%J.stdout
#BSUB -eo ${log_dir}/${group}_${level}_%J.stderr
#BSUB -L /bin/bash

ml purge
ml anaconda3 
source activate cutie

python $work_path/../scripts/calculate_cutie.py -i $file_path

EOF
        bsub < $job_path
    done
done
