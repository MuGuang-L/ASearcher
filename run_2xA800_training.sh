#!/bin/bash

# ==============================================================================
# ASearcher Training Script for 2x A800 (160GB) Setup
# Task: 10-turn max, 3B/7B Model Full Parameter Fine-tuning
# ==============================================================================

# 1. Export your API keys for the search environment
export SERPER_API_KEY="YOUR_SERPER_API_KEY"
export JINA_API_KEY="YOUR_JINA_API_KEY"

# 2. Define Experiment Info
EXPERIMENT_NAME="asearcher-3b-web-10turns"
TRIAL_NAME="run1"

# 3. Model & Data Paths
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"   # Or Qwen/Qwen2.5-7B-Instruct
DATA_PATH="path_to_training_data"       # Replace with your local dataset path

# 4. Launch the training via AReaL local launcher
#    - allocation_mode: sglang.d1p1t1+d1c4 
#      This means 1 GPU for SGLang (Rollout) and 1 GPU for FSDP (Trainer)
#    - max_turns: 10 (Strictly limiting to 10 turns to save KV cache and protect the small model)
python3 -m areal.launcher.local ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_web.yaml \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=${TRIAL_NAME} \
    actor.path=${MODEL_PATH} \
    train_dataset.path=${DATA_PATH} \
    allocation_mode="sglang.d1p1t1+d1" \
    max_turns=10 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=2

echo "Training launched for $EXPERIMENT_NAME"
