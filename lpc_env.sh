#!/usr/bin/env bash
NAME=hgcalKeras
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh

source $LCG
python -m venv --copies $NAME
source $NAME/bin/activate
export PYTHONPATH=$PWD/$NAME/lib/python3.6/site-packages:$PYTHONPATH
