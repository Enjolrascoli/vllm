env CUDA_VISIBLE_DEVICES=5 \
    VLLM_BELADY=0 \
    vllm serve Qwen/Qwen3-8B \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 64 \
        --port 40002
