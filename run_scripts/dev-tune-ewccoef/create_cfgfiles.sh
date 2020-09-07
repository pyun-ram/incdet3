#!/bin/bash
# for ewc_coef in 0.01 0.1 1 10 100; do
# for cv in 0 1 2; do
# python3 tools/create_config_files.py \
#     --template-cfg-path configs/tune-ewccoef/kdewc-xanchor-cvx.py \
#     --save-path configs/tune-ewccoef/kdewc-${ewc_coef}anchor-cv${cv}.py \
#     --key-pairs AcvA=${cv},Aewc_coefA=${ewc_coef}
# done
# done

for ewc_coef in 0.01 0.1 1 10 100; do
for cv in 0 1 2; do
python3 tools/create_config_files.py \
    --template-cfg-path configs/tune-ewccoef/ewc-xanchor-cvx.py \
    --save-path configs/tune-ewccoef/ewc-${ewc_coef}anchor-cv${cv}.py \
    --key-pairs AcvA=${cv},Aewc_coefA=${ewc_coef}
done
done