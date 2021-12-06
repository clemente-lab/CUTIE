#!/bin/bash

cwd=$(pwd)

cd ~/Code/cutie
source activate cutie
# python scripts/calculate_cutie.py -i $cwd/matt_config.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvV/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/VvA/gene-taxa-config-relab.ini

'''
python scripts/calculate_cutie.py -i $cwd/configs-new/CvA/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvV/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/VvA/gene-Trp-EAE.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvV/gene-Trp-both.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvA/gene-Trp-both.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/CvA/gene-Trp-EAE.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvC/gene-Trp-both.ini

python scripts/calculate_cutie.py -i $cwd/configs-new/CvA/gene-bile-config.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/CvA/gene-taxa-config-absab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/CvA/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/CvA/gene-Trp-EAE.ini

python scripts/calculate_cutie.py -i $cwd/configs-new/NvA/gene-bile-config.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvA/gene-taxa-config-absab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvA/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvA/gene-Trp-both.ini

python scripts/calculate_cutie.py -i $cwd/configs-new/NvC/gene-bile-config.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvC/gene-taxa-config-absab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvC/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvC/gene-Trp-both.ini

python scripts/calculate_cutie.py -i $cwd/configs-new/NvV/gene-bile-config.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvV/gene-taxa-config-absab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvV/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/NvV/gene-Trp-both.ini

python scripts/calculate_cutie.py -i $cwd/configs-new/VvA/gene-bile-config.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/VvA/gene-taxa-config-absab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/VvA/gene-taxa-config-relab.ini
python scripts/calculate_cutie.py -i $cwd/configs-new/VvA/gene-Trp-EAE.ini
'''

cd $cwd
