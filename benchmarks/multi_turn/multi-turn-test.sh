#!/bin/bash

python3 benchmark_serving_multi_turn.py \
	--input-file ./sharegpt_conv_128.json \
	--output-file multi-turn-output.json \
	--seed 12345 \
	--model Qwen/Qwen3-8B \
	--url http://localhost:40002 \
	--request-rate 5 \
	--verbose \
	--excel-output \
