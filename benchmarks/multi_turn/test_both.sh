#!/bin/bash

# Kill background jobs on Ctrl-C (SIGINT) or kill (SIGTERM)
trap 'kill $pid1 $pid2 2>/dev/null; exit 1' SIGINT SIGTERM

trace_file=${1:-100_trace.jsonl}
max_tokens=4096
model=Qwen/Qwen3-8B

curl -X POST 'http://localhost:40001/reset_prefix_cache'  
curl -X POST 'http://localhost:40002/reset_prefix_cache'  

python3 replay_trace_stat.py \
    --trace-file "$trace_file" \
    --model "$model" \
    --max-tokens "$max_tokens" \
    --output-prefix "[lru]_" \
    --port 40001 &
pid1=$!

python3 replay_trace_stat.py \
    --trace-file "$trace_file" \
    --model "$model" \
    --max-tokens "$max_tokens" \
    --output-prefix "[belady]_" \
    --port 40002 &
pid2=$!

wait $pid1 $pid2

#python3 process_stats.py 