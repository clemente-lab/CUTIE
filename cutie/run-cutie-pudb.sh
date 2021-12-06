#!/bin/bash

cwd=$(pwd)

cd ~/Code/cutie
conda activate cutie
python -m pudb scripts/calculate_cutie.py -i $cwd/tutorial_config.ini

cd $cwd
