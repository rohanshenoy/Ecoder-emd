#!/usr/bin/env bash
NAME=hgcalKeras
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh

source $LCG
# following https://aarongorka.com/blog/portable-virtualenv/, an alternative is https://github.com/pantsbuild/pex
python -m venv --copies $NAME
source $NAME/bin/activate
python -m pip install setuptools pip --upgrade
python -m pip install tensorflow==1.5
#python -m pip install keras==2.3.1
python -m pip install pyjet 

export PYTHONPATH=$PWD/$NAME/lib/python3.6/site-packages:$PYTHONPATH


sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*
sed -i "2a source ${LCG}" $NAME/bin/activate
sed -i "3a export PYTHONPATH=${PWD}/${NAME}/lib/python3.6/site-packages:$PYTHONPATH" $NAME/bin/activate
ipython kernel install --user --name=$NAME
tar -zcf ${NAME}.tar.gz ${NAME}
