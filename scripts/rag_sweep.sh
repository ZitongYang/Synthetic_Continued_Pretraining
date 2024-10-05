#!/bin/bash
#SBATCH --requeue
#SBATCH --exclude=XXX
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --open-mode=append
#SBATCH --partition=XXX
#SBATCH --time=14-0
#SBATCH --job-name=XXX
#SBATCH --account=XXX
#SBATCH --array=0-120%4

CHUNKSIZE=(256 512 1024)
CHUNKORDER=(best_last best_first)
RERANKTOPK=(1 2 4 8 16)
MODELPATH=(/path/to/entigraph meta-llama/Meta-Llama-3-8B)
TEMP=(0.1 0.3 0.5 0.7)

# Calculate the indices for each parameter based on SLURM_ARRAY_TASK_ID
index1=$(( SLURM_ARRAY_TASK_ID / (${#CHUNKORDER[@]} * ${#RERANKTOPK[@]} * ${#MODELPATH[@]} * ${#TEMP[@]}) ))
index2=$(( (SLURM_ARRAY_TASK_ID / (${#RERANKTOPK[@]} * ${#MODELPATH[@]} * ${#TEMP[@]})) % ${#CHUNKORDER[@]} ))
index3=$(( (SLURM_ARRAY_TASK_ID / (${#MODELPATH[@]} * ${#TEMP[@]})) % ${#RERANKTOPK[@]} ))
index4=$(( (SLURM_ARRAY_TASK_ID / ${#TEMP[@]}) % ${#MODELPATH[@]} ))
index5=$(( SLURM_ARRAY_TASK_ID % ${#TEMP[@]} ))

# Retrieve the corresponding parameter values
chunk_size=${CHUNKSIZE[$index1]}
chunk_order=${CHUNKORDER[$index2]}
rerank_top_k=${RERANKTOPK[$index3]}
model_path=${MODELPATH[$index4]}
temp=${TEMP[$index5]}

# Print the configuration for sanity check
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Chunk Size: $chunk_size"
echo "Chunk Order: $chunk_order"
echo "Rerank Top K: $rerank_top_k"
echo "Model Path: $model_path"
echo "Temperature: $temp"

srun python evaluation.py --eval_func=eval_quality_qa_with_rag \
    --model_path=$model_path \
    --eval_temperature=$temp \
    --embedding_model_path=text-embedding-3-large \
    --text_split_strategy=recursive \
    --chunk_size=$chunk_size \
    --chunk_overlap=0 \
    --retrieval_max_k=128 \
    --retrieval_top_k=128 \
    --rerank_model_path=rerank-english-v3.0 \
    --rerank_top_k=$rerank_top_k \
    --retrieved_chunk_order=$chunk_order
