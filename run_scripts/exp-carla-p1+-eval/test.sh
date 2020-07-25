GPUID="0"
task=task3

for cv in {0..4} ; do
for reuse_tag in "reuse" "noreuse" ; do
for domain in "domain1" "domain2" "domain3" ; do
for mode in "finetuning" "jointtraining" "lwf" ; do
./run_scripts/clear_gpus.sh
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-eval-${mode}-${task}-${domain}-${reuse_tag}-${cv} \
    --cfg-path configs/exp-carla-p1+-eval/${mode}_${task}_${domain}_${reuse_tag}_${cv}.py \
    --mode test
done
done
done
done
done

GPUID="0"
task=task2

for cv in {0..4} ; do
for reuse_tag in "reuse" "noreuse" ; do
for domain in "domain1" "domain2" ; do
for mode in "finetuning" "jointtraining" "lwf" ; do
./run_scripts/clear_gpus.sh
CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
    --tag July24-expcarlap1+-eval-${mode}-${task}-${domain}-${reuse_tag}-${cv} \
    --cfg-path configs/exp-carla-p1+-eval/${mode}_${task}_${domain}_${reuse_tag}_${cv}.py \
    --mode test
done
done
done
done
done