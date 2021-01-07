
~


#!/bin/bash

#Run script for  ADP
#Artificial Intelligence (CS 486)

if [[ ! -d 'venv' ]]; then
    rm -rf venv
    python3.7 -m venv ./venv --without-pip
    source ./venv/bin/activate
    curl https://bootstrap.pypa.io/get-pip.py | python
    deactivate
    source ./venv/bin/activate
    pip install numpy
fi

python3.7 ADP.py
