#!/bin/bash

cwd=$(pwd)

cd ~/Code/anapi2
source activate anapi
python -m pudb scripts/table-analysis.py -t $cwd/inputs -o $cwd/inputs-modified-test -ao 'long-to-wide'


cd $cwd
