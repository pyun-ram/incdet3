#!/bin/bash
# # after il (targetdataset/sourcedataset) (best/last model)
# for dataset_tag in "sourcedataset"
# do
# for model_tag in "best" "last"
# do
# for i in {0..4}
# do
# echo $dataset_tag $model_tag $i
# python3 main.py \
#     --tag lwf_distloss_cv$i-afteril-$dataset_tag-$model_tag \
#     --cfg-path configs/exp-carla-eval/lwf_distloss_cv$i-afteril-$dataset_tag-$model_tag.py \
#     --mode test

# python3 main.py \
#     --tag jointtraining_cv$i-afteril-$dataset_tag-$model_tag \
#     --cfg-path configs/exp-carla-eval/jointtraining_cv$i-afteril-$dataset_tag-$model_tag.py \
#     --mode test

# python3 main.py \
#     --tag finetuning_cv$i-afteril-$dataset_tag-$model_tag \
#     --cfg-path configs/exp-carla-eval/finetuning_cv$i-afteril-$dataset_tag-$model_tag.py \
#     --mode test
# done
# done
# done

#before il (targetdataset/sourcedataset)
python3 main.py \
    --tag beforeil-targetdataset \
    --cfg-path configs/exp-carla-eval/beforeil-targetdataset.py \
    --mode test

python3 main.py \
    --tag beforeil-sourcedataset \
    --cfg-path configs/exp-carla-eval/beforeil-sourcedataset.py \
    --mode test