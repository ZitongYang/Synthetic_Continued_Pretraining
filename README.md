# Synthetic Continued Pretraining

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/ZitongYang/Synthetic_Continued_Pretraining/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)

This repository contains the code for the paper [Synthetic Continued Pretraining](https://arxiv.org/pdf/2409.07431).

## Overview

This codebase implements the entire pipeline for synthetic continued pretraining using the EntiGraph synthetic data generator. It includes:

- Code for generating synthetic data with EntiGraph
- Scripts for continued pretraining with Llama 3 8B
- Evaluation tools for the continually pretrained model
- Instruction tuning process
- Interactive chatbot based on the instruction-tuned model

## Table of Contents

1. [Installation](#installation)
2. [EntiGraph Synthetic Continued Pretraining](#entigraph-synthetic-continued-pretraining)
   - [Step 1: EntiGraph Synthetic Data Generation](#step-1-entigraph-synthetic-data-generation)
   - [Step 2: Tokenization](#step-2-tokenization)
   - [Step 3: Downloading and Tokenizing Replay Data](#step-3-downloading-and-tokenizing-replay-data)
   - [Step 4: Continued Pretraining](#step-4-continued-pretraining)
   - [Step 5: Evaluation on QuALITY QA Set](#step-5-evaluation-on-quality-qa-set)
3. [Instruction Tuning on Continued Pretrained Model](#instruction-tuning-on-continued-pretrained-model)
   - [Step 1: Downloading and Tokenizing the Instruction Tuning Data](#step-1-downloading-and-tokenizing-the-instruction-tuning-data)
   - [Step 2: Instruction Tuning](#step-2-instruction-tuning)
   - [Step 3: Hosting the Chatbot](#step-3-hosting-the-chatbot)
4. [Citation](#citation)

## Installation

```bash
git clone https://github.com/ZitongYang/Synthetic_Continued_Pretraining.git
cd Synthetic_Continued_Pretraining
pip install -r requirements.txt
huggingface-cli login --token <huggingface token>; # required, you need this to access Llama 3 pretrained weights
wandb login <weights and bias token>; # optional, ignore if you don't want to log your training process
```

## EntiGraph Synthetic Continued Pretraining

Our experiments use the [QuALITY dataset](https://arxiv.org/abs/2112.08608) as the source documents.

### Step 1: EntiGraph Synthetic Data Generation

1. Set your OpenAI API key in `data/dataset/openai.key`.
2. To run the EntiGraph procedure for the `i`-th article using `gpt-4-turbo`:

```bash
python data/entigraph.py i
```

The resulting synthetic data will be saved in `data/dataset/raw/quality_entigraph_gpt-4-turbo/`.

### Step 2: Tokenization

Tokenize the EntiGraph synthetic data:

```bash
mkdir -p data/dataset/bins/
python data/tokenize_entigraph.py
```

This will save the resulting binary files in `data/dataset/bins/quality_all-graphgpt-4-turbo.bin`.

### Step 3: Downloading and Tokenizing Replay Data

Download and tokenize 1B tokens of RedPajama dataset as replay data:

```bash
python data/tokenize_redpj.py
```

This will save two binary files:
- `data/dataset/bins/togethercomputer_RedPajama_Data_1T_Sample_None_train.bin`
- `data/dataset/bins/togethercomputer_RedPajama_Data_1T_Sample_None_test.bin`

To inspect the synthetic data generated:

```bash
python data/cptdata.py
```

### Step 4: Continued Pretraining

To perform continued pretraining on Llama 3 8B using the EntiGraph synthetic data:

```bash
chmod 777 scripts/train.sh
./scripts/train.sh \
    --lr 5e-06 \
    --rr 0.1 \
    --epochs 2 \
    --bs 16 \
    --wd 0.01 \
    --warmup 0.05 \
    --task_name quality
```

Arguments:
- `--lr`: Peak learning rate
- `--rr`: RedPajama replay rate
- `--epochs`: Total epochs to run
- `--bs`: Batch size
- `--wd`: Weight decay factor
- `--task_name`: Dataset choice (`quality` for EntiGraph synthetic data, `instruct` for UltraChat instruction tuning data)

The resulting checkpoint will be saved under `ckpts/quality-lr5e-06-rr0.1-epochs2-bs16-wd0.01-warmup0.05-MetaLlama38B`.

### Step 5: Evaluation on QuALITY QA Set

To evaluate on the QuALITY QA set:

```bash
python evaluation.py --model_path=ckpts/quality-lr5e-06-rr0.1-epochs2-bs16-wd0.01-warmup0.05-MetaLlama38B
```

The output will be stored in `out/qualityqa-quality-lr5e-06-rr0.1-epochs2-bs16-wd0.01-warmup0.05-MetaLlama38B.json`.

To parse the output into accuracy metrics, refer to `notebooks/nb_qa_eval.ipynb`.

## Instruction Tuning on Continued Pretrained Model

### Step 1: Downloading and Tokenizing the Instruction Tuning Data

We use the [UltraChat dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) and Llama 3.1 Instruct chat template:

```bash
python data/tokenize_instruct.py
```

This will save the instruction tuning data in `data/dataset/bins/ultrachat_train.bin` and `data/dataset/bins/ultrachat_test.bin`.

### Step 2: Instruction Tuning

To perform instruction tuning on the continually pretrained model:

```bash
./scripts/train.sh \
    --lr 5e-06 \
    --rr 0.1 \
    --epochs 2 \
    --bs 128 \
    --wd 0.01 \
    --warmup 0.05 \
    --task_name instruct \
    --model_name ckpts/quality-lr5e-06-rr0.1-epochs2-bs16-wd0.01-warmup0.05-MetaLlama38B
```

The checkpoint will be saved in `ckpts/instruct-lr5e-06-rr0.1-epochs2-bs128-wd0.01-warmup0.05-qualitylr5e06rr0.1epochs2bs16wd0.01warmup0.05MetaLlama38B`.

### Step 3: Hosting the Chatbot

To launch an interactive session with the instruction-tuned EntiGraph model:

```bash
python interactive.py
```

You can ask questions about QuALITY articles (e.g., Tell me about the article "defining decay down".).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{yang2024syntheticcontinuedpretraining,
      title={Synthetic continued pretraining}, 
      author={Zitong Yang and Neil Band and Shuangping Li and Emmanuel Candès and Tatsunori Hashimoto},
      year={2024},
      eprint={2409.07431},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.07431}, 
}
```