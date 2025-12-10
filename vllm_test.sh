#!/bin/bash
        #--tensor-parallel-size 2 \
export VLLM_SERVER_DEV_MODE=1

env CUDA_VISIBLE_DEVICES=4 \
    VLLM_BELADY=0 \
    vllm serve Qwen/Qwen3-8B \
        --port 40001 \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 64 &

env CUDA_VISIBLE_DEVICES=5 \
    VLLM_BELADY=1 \
    vllm serve Qwen/Qwen3-8B \
        --port 40002 \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 64 &