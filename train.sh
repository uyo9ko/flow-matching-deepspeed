#!/bin/bash

# Data and output directories
DATA_DIR="imagenet-1k/data"
CKPT_DIR="checkpoints"
SAMPLES_DIR="samples"
WANDB_PROJECT_DIR="logs"

# Training configuration
MIXED_PRECISION="fp16"  # Options: no, fp16, bf16, fp8
CHECKPOINTING_STEPS=5000  # Or specify a number
SAMPLING_STEPS=1000

# Optional features (uncomment to enable)
ADDITIONAL_ARGS=""
#ADDITIONAL_ARGS="$ADDITIONAL_ARGS --fp16"
# ADDITIONAL_ARGS="$ADDITIONAL_ARGS --use_stateful_dataloader"
#ADDITIONAL_ARGS="$ADDITIONAL_ARGS --resume_from_checkpoint path/to/checkpoint"

# Accelerate launch command
accelerate launch \
    --multi_gpu \
    --mixed_precision "$MIXED_PRECISION" \
    trainer_deepspeed.py \
        --data_dir "$DATA_DIR" \
        --ckpt_dir "$CKPT_DIR" \
        --samples_dir "$SAMPLES_DIR" \
        --wandb_project_dir "$WANDB_PROJECT_DIR" \
        --mixed_precision "$MIXED_PRECISION" \
        --checkpointing_steps "$CHECKPOINTING_STEPS" \
        --sampling_steps "$SAMPLING_STEPS" \
        $ADDITIONAL_ARGS
