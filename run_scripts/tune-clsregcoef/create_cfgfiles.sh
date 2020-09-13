#!/bin/bash
for reg2coef in 1; do
for clsregcoef in 0.01 0.1 1 10 100; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-clsregcoef/ewc-cv-clsregcoef-template.py \
    --save-path configs/tune-clsregcoef/ewc-cv${cv}-${clsregcoef}.py \
    --key-pairs TBDcvTBD=${cv},TBDreg2coefTBD=${reg2coef},TBDclsregcoefTBD=${clsregcoef}
done
done
done

for reg2coef in 1; do
for clsregcoef in 0.01 0.1 1 10 100; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-clsregcoef/kdewc-cv-clsregcoef-template.py \
    --save-path configs/tune-clsregcoef/kdewc-cv${cv}-${clsregcoef}.py \
    --key-pairs TBDcvTBD=${cv},TBDreg2coefTBD=${reg2coef},TBDclsregcoefTBD=${clsregcoef}
done
done
done