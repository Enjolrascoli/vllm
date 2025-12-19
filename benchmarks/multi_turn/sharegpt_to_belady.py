#!/usr/bin/env python3
"""
sharegpt_to_vllm_multiturn.py
Creates a deterministic multi-turn trace compatible with
vllm/benchmarks/benchmark_serving_multi_turn.py
Fields per row:  chat_id  prompt  timestamp  parent_chat_id
"""
import argparse, datetime as dt, hashlib, itertools, json, pathlib, random, sys, time
import tiktoken
from transformers import AutoTokenizer
import numpy as np
import os

SECONDS_PER_OUTPUT_TOKEN = 0.05   # 20 tokens/s  – change if you like
TOKEN_PER_SECOND = 1000
RNG = random.Random(12345)


def build_trace(in_file: pathlib.Path, out_file: pathlib.Path,
                max_conv: int, request_rate: float, max_turn: int, turn_gap: float, model: str):
    data = json.loads(in_file.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model) 
    os.makedirs("trace", exist_ok=True)
    
    # keep only valid multi-turn dialogs
    convs = []
    for raw in data:
        turns = [{"from": m["from"], "value": m["value"]} for m in raw["conversations"]]
        if len(turns) % 2 == 0 and len(turns) >= 2:
            convs.append({"id": raw["id"], "turns": turns})
    RNG.shuffle(convs)
    convs = convs[:max_conv]

    # global Poisson process for *first* user message of each conversation
    t0 = time.time()
    next_conv = 0.0
    MIN_GAP = turn_gap                     # seconds between parent and child
    traces = []
    
    turn_rate = request_rate / 10

    for i, conv in enumerate(convs):
        next_turn = next_conv
        # cumulative chat history in OpenAI format
        # history = [{"role": "system", "content": "You are a helpful assistant."}]
        turn_idx = 0
        for msg in conv["turns"]:
            if turn_idx > max_turn: break
            role = "user" if msg["from"] == "human" else "assistant"
            # history.append({"role": role, "content": msg["value"]})
            if role == "user":          # we only *request* the model when user speaks
                #prompt_txt = "\n".join(f"{h['role']}: {h['content']}" for h in history)
                prompt = msg["value"]

                parent_chat_id = f"{conv['id']}_turn{turn_idx-2}" if turn_idx > 0 else ""
                chat_id = f"{conv['id']}_turn{turn_idx}"

                traces.append({
                    "conv_id": chat_id,
                    "timestamp": next_turn,
                    "prompt": prompt,
                    "parent_chat_id": parent_chat_id
                })

                # timestamp for next turn
                # gap = MIN_GAP + (tokens_of_previous_assistant / TOKEN_PER_SECOND)
                # gap = MIN_GAP
                gap = 0.0
                if turn_idx > 0:
                    # TODO: process chat template
                    prev_assistant_txt = conv["turns"][turn_idx-1]["value"]
                    gap += len(tokenizer.encode(prev_assistant_txt)) / TOKEN_PER_SECOND
                gap = max(gap, MIN_GAP)
                gap += np.random.exponential(1.0 / turn_rate) # add poisson gap
                next_turn += gap
            turn_idx += 1

        # Poisson gap between *conversations*
        delta = np.random.exponential(1.0 / request_rate)
        next_conv += delta

    traces.sort(key=lambda x: x["timestamp"]) # sort by timestamp
    conv_to_req = {tr["conv_id"]: idx for idx, tr in enumerate(traces)}
    for i, trace in enumerate(traces):
        trace["req_id"] = i
        trace["parent_req_id"] = conv_to_req.get(trace["parent_chat_id"], -1)
    out_file.write_text("\n".join(json.dumps(t) for t in traces))
    print(f"Wrote {len(traces)} rows -> {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, default="ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--output", type=pathlib.Path, default="sharegpt_multiturn_trace.jsonl")
    ap.add_argument("--max-conversations", type=int, default=1000)
    ap.add_argument("--max-turn", type=int, default=10)
    ap.add_argument("--request-rate", type=float, default=4.0, help="Average conversation-start rate")
    ap.add_argument("--turn-gap", type=float, default=10.0, help="Average conversation-start rate")
    ap.add_argument("-m", "--model", type=str, default="Qwen/Qwen3-1.7B", help="Path of the LLM model")
    args = ap.parse_args()
    build_trace(args.input, args.output, args.max_conversations, args.request_rate, args.max_turn, args.turn_gap, args.model)