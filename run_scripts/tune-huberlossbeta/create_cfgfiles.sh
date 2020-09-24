#!/bin/bash
for beta in 1e-7 1e-6 1e-5 1e-4 1e-3; do
for ewc_coef in 0.01; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-huberlossbeta/kdewc-beta-cv-template.py \
    --save-path configs/tune-huberlossbeta/kdewc-${beta}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDewc_coefTBD=${ewc_coef},TBDbetaTBD=${beta}
done
done
done

for beta in 1e-7 1e-6 1e-5 1e-4 1e-3; do
for ewc_coef in 0.01; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-huberlossbeta/ewc-beta-cv-template.py \
    --save-path configs/tune-huberlossbeta/ewc-${beta}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDewc_coefTBD=${ewc_coef},TBDbetaTBD=${beta}
done
done
done

for beta in 1e-7 1e-6 1e-5 1e-4 1e-3; do
for ewc_coef in 0.01; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-huberlossbeta/pseudoewc-beta-cv-template.py \
    --save-path configs/tune-huberlossbeta/pseudoewc-${beta}-cv${cv}.py \
    --key-pairs TBDcvTBD=${cv},TBDewc_coefTBD=${ewc_coef},TBDbetaTBD=${beta}
done
done
done