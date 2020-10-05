#!/bin/bash
for anchorsample in all biased32 biased64 biased128; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-anchorsample/mas-anchorsample-cv-template.py \
    --save-path configs/tune-anchorsample/mas-${anchorsample}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDanchorsampleTBD=${anchorsample}
done
done

for anchorsample in all biased32 biased64 biased128; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-anchorsample/kdmas-anchorsample-cv-template.py \
    --save-path configs/tune-anchorsample/kdmas-${anchorsample}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDanchorsampleTBD=${anchorsample}
done
done