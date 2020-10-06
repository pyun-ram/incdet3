#!/bin/bash
for regcoef in 0.3 1 3 10 30; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-regcoef/mas-regcoef-cv-template.py \
    --save-path configs/tune-regcoef/mas-${regcoef}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDregcoefTBD=${regcoef}

python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-regcoef/kdmas-regcoef-cv-template.py \
    --save-path configs/tune-regcoef/kdmas-${regcoef}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDregcoefTBD=${regcoef}

done
done
