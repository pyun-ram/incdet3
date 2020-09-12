#!/bin/bash
for reg2coef in 0.01 0.1 1 10 100; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-reg2coef/ewc-cv-reg2coef-template.py \
    --save-path configs/tune-reg2coef/ewc-cv${cv}-${reg2coef}.py \
    --key-pairs TBDcvTBD=${cv},TBDreg2coefTBD=${reg2coef}
done
done

for reg2coef in 0.01 0.1 1 10 100; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-reg2coef/kdewc-cv-reg2coef-template.py \
    --save-path configs/tune-reg2coef/kdewc-cv${cv}-${reg2coef}.py \
    --key-pairs TBDcvTBD=${cv},TBDreg2coefTBD=${reg2coef}
done
done