#!/bin/bash
for mascoef in 0.01 0.1 1 10 100; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-mascoef/mas-mascoef-cv-template.py \
    --save-path configs/tune-mascoef/mas-${mascoef}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDmascoefTBD=${mascoef}

python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-mascoef/kdmas-mascoef-cv-template.py \
    --save-path configs/tune-mascoef/kdmas-${mascoef}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDmascoefTBD=${mascoef}

done
done
