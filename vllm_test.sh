#!/bin/bash
        #--tensor-parallel-size 2 \
trap 'kill $pid1 $pid2 2>/dev/null; exit 1' SIGINT SIGTERM

export VLLM_SERVER_DEV_MODE=1
#export VLLM_LOGGING_LEVEL=DEBUG
#export VLLM_TRACE_FUNCTION=1

env CUDA_VISIBLE_DEVICES=4 \
    VLLM_BELADY=0 \
    vllm serve Qwen/Qwen3-1.7B \
        --port 40001 \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --enable-request-id-headers \
        --gpu-memory-utilization 0.90 \
        --chat-template min_chat_template.jinja \
        --enable-prompt-tokens-details 2> 40001.log &
pid1=$!

env CUDA_VISIBLE_DEVICES=5 \
    VLLM_BELADY=1 \
    vllm serve Qwen/Qwen3-1.7B \
        --port 40002 \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --enable-request-id-headers \
        --gpu-memory-utilization 0.90 \
        --chat-template min_chat_template.jinja \
        --enable-prompt-tokens-details 2> 40002.log &
pid2=$!

wait $pid1 $pid2