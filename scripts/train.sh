#!/bin/bash
# setting up default hyperparameters
subsample_ratio=1.0 # change this parameter to run the scaling plot
model_name="meta-llama/Meta-Llama-3-8B"
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --rr) rr="$2"; shift 2;;
        --epochs) epochs="$2"; shift 2 ;;
        --bs) bs="$2"; shift 2 ;;
        --wd) wd="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --task_name) task_name="$2"; shift 2 ;;
        --subsample_ratio) subsample_ratio="$2"; shift 2 ;;
        --model_name) model_name="$2"; shift 2 ;;
        --run_eval) run_eval=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done
gpu_count=$(nvidia-smi -L | wc -l)
pretty_name=${model_name##*/}
pretty_name=$(echo "$pretty_name" | sed 's/-//g')
grad_acc=$((bs / 8))
# Then, use an if-else statement to set the run_name
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
echo "Running experiment with run name: $run_name"
output_dir="ckpts/${run_name}"

# Execute the training command with the specific hyperparameters
torchrun --nproc_per_node=$gpu_count  train.py \
    --model_name=$model_name \
    --block_size=2048 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=3 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --rehersal_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --task_name=$task_name \
    --logging_steps=1 \
    --run_name=$run_name \
    --bf16=True \
    --output_dir=$output_dir \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --evaluation_strategy="no" \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --log_level="info" \
    --fsdp="hybrid_shard auto_wrap" \
    --fsdp_config="scripts/config/fsdp_config.json"