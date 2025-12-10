#!/usr/bin/env python3
"""Replay trace requests produced by sharegpt_to_belady.py against a vLLM server."""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import aiohttp

from vllm.v1.core.sched.future_usage import FutureUsageMap
from benchmark_serving_multi_turn import send_request, send_turn, ServerResponse, remove_prefix, get_messages_token_count, get_token_count, RequestStats
from transformers import AutoTokenizer
import pandas as pd
from datetime import datetime, timezone, timedelta
import requests, prometheus_client.parser


@dataclass
class TraceEntry:
    conv_id: str
    base_id: str
    prompt: str
    timestamp: float
    req_id: int
    parent_req_id: int

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TraceEntry":
        conv_id = data["conv_id"]
        base_id = conv_id.split("_turn")[0]
        return cls(
            conv_id=conv_id,
            base_id=base_id,
            prompt=data["prompt"],
            timestamp=float(data["timestamp"]),
            req_id=int(data["req_id"]),
            parent_req_id=int(data.get("parent_req_id", -1)),
        )


# get prefix hit and query counts
def snapshot(chat_url: str) -> tuple[float, float]:
    url = chat_url.replace("/v1/chat/completions", "/metrics")
    text = requests.get(url, timeout=5).text
    families = list(prometheus_client.parser.text_string_to_metric_families(text))
    metrics = {m.name: m for m in families}
    # counters are the first (and only) sample in each family
    queries = metrics["vllm:prefix_cache_queries"].samples[0].value
    hits    = metrics["vllm:prefix_cache_hits"].samples[0].value
    return float(queries), float(hits)


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


async def handle_entry(
    entry: TraceEntry,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    history: List[Dict[str, str]],
    stats: List[RequestStats],
    completion_events: Dict[int, asyncio.Event],
    start_time: float,
    tokenizer: AutoTokenizer,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        event = completion_events[entry.req_id]
        min_tokens = None if args.min_tokens < 0 else args.min_tokens
        max_tokens = None if args.max_tokens < 0 else args.max_tokens
        send_started = datetime.now()
        response: ServerResponse = await send_request(
            session,
            messages,
            args.url,
            args.model,
            max_tokens=max_tokens,
            min_tokens=min_tokens
        )
    
    if response.valid is False:
        # Request failed
        return None

    # Compute number of tokens in input / output
    input_num_tokens = get_messages_token_count(tokenizer, messages)

    # Num tokens in the user's last question
    question_num_tokens = get_token_count(tokenizer, messages[-1]["content"])

    # Num tokens in the history/context of the question
    assert input_num_tokens >= question_num_tokens
    history_num_tokens = input_num_tokens - question_num_tokens

    # Num tokens in the LLM's answer (first chunk and full answer)
    first_chunk_tokens = get_token_count(tokenizer, response.first_chunk)

    output_content = response.content
    output_num_tokens = get_token_count(tokenizer, output_content)

    # Prefix caching approximated cached percent
    approx_cached_percent = (
        100.0 * (history_num_tokens / input_num_tokens) if input_num_tokens > 0 else 0.0
    )

    # Compute the correct TTFT and TPOT (based on tokens and not chunks).
    # Required because multiple output tokens may be bundled in a single chunk.
    if output_num_tokens > 1 and output_num_tokens > first_chunk_tokens:
        # More than one token and more than one chunk in the output
        decode_ms = response.latency_ms - response.ttft_ms
        decode_num_tokens = output_num_tokens - first_chunk_tokens
        tpot_ms = decode_ms / decode_num_tokens
    else:
        # In this case: output_num_tokens == first_chunk_tokens
        # Output was a single chunk (output_num_tokens > 1)
        # or even a single token (output_num_tokens == 1)
        tpot_ms = 0.0

    if first_chunk_tokens > 1:
        # First chunk had multiple tokens, adjust TTFT for a single token
        delta_ms = (first_chunk_tokens - 1) * tpot_ms
        ttft_ms = max(0.1, response.ttft_ms - delta_ms)
    else:
        # First chunk had only one token
        ttft_ms = response.ttft_ms

    rs = RequestStats(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=response.latency_ms,
        start_time_ms=response.start_time_ms,
        input_num_turns=len(messages),
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        output_num_chunks=response.num_chunks,
        output_num_first_chunk_tokens=first_chunk_tokens,
        approx_cached_percent=approx_cached_percent,
        conversation_id=entry.conv_id,
        client_id=1,
    )
    stats.append(rs)

    # update conversation
    history.extend(
        [
            {"role": "user", "content": entry.prompt},
            {"role": "assistant", "content": response.content},
        ]
    )

    # mark the request finished
    event.set()


async def replay_traces(entries: List[TraceEntry], args: argparse.Namespace) -> None:
    histories: Dict[str, List[Dict[str, str]]] = {}
    completion_events: Dict[int, asyncio.Event] = {}
    pending: List[asyncio.Task] = []
    stats: List[RequestStats] = []
    
    FutureUsageMap.compute(entries)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        for entry in entries:
            completion_events.setdefault(entry.req_id, asyncio.Event())

            scheduled_time = start_time + entry.timestamp
            delay = scheduled_time - time.time()
            if delay > 0 and args.follow_timestamp:
                await asyncio.sleep(delay)

            if entry.parent_req_id != -1:
                parent_event = completion_events.get(entry.parent_req_id)
                if parent_event:
                    await parent_event.wait()

            history = histories.setdefault(entry.base_id, [])
            messages = list(history)
            messages.append({"role": "user", "content": entry.prompt})

            print(f"sending req_id={entry.req_id} at {time.time() - start_time:.2f}s", flush=True)

            task = asyncio.create_task(
                handle_entry(
                    entry,
                    session,
                    args,
                    messages,
                    history,
                    stats,
                    completion_events,
                    start_time,
                    tokenizer,
                    semaphore,
                )
            )
            pending.append(task)

        if pending:
            await asyncio.gather(*pending)
            raw_data = pd.DataFrame(stats)
            
            duration = time.time() - start_time
            timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%m%d_%H%M")
            csv_filename = f"{args.output_prefix}replay_trace_stats_{timestamp}.csv"
            raw_data.to_csv(csv_filename, index=False)
            print(f"Wrote replay stats for {len(raw_data)} requests to {csv_filename}", flush=True)
            
            after_num_cache_queries, after_num_cache_hits = snapshot(args.url)
            prefix_cache_hit_ratio = after_num_cache_hits / after_num_cache_queries if after_num_cache_queries > 0 else 0.0
            print(f"Prefix cache hit ratio: {prefix_cache_hit_ratio:.2f}", flush=True)
            
            if args.output_prefix:
                instance = "lru" if "lru" in args.output_prefix else "belady"
            elif args.port == 40001:
                instance = "lru"
            elif args.port == 40002:
                instance = "belady"

            summary = { 
                "instance": instance,
                "prefix_cache_hit_ratio": prefix_cache_hit_ratio,
                "total_requests": len(stats),
                "model": args.model,
                "timestamp": timestamp,
                "cache_hits": after_num_cache_hits,
                "cache_queries": after_num_cache_queries,
                "duration": f"{duration:.2f}s"
            }
            json_filename = f"{args.output_prefix}summary_{timestamp}.json"
            with open(json_filename, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Wrote summary stats to {json_filename}", flush=True)
            

def load_trace(trace_file: str) -> List[TraceEntry]:
    entries: List[TraceEntry] = []
    with open(trace_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(TraceEntry.from_json(data))

    entries.sort(key=lambda e: (e.timestamp, e.req_id))
    return entries


async def main(args: argparse.Namespace) -> None:
    entries = load_trace(args.trace_file)
    if not entries:
        print("Trace file is empty; nothing to replay.")
        return

    print(f"Loaded {len(entries)} trace entries from {args.trace_file}")
    await replay_traces(entries, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay trace against vLLM service")
    parser.add_argument("--trace-file", type=str, required=True, help="Path to the trace JSONL file")
    parser.add_argument( "--url", type=str, default=None, help="vLLM service URL")
    parser.add_argument("--port", type=int, default=40001, help="Port for the vLLM service")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--max-concurrent-requests", type=int, default=25, help="Max concurrent requests")
    parser.add_argument("--follow-timestamp", type=bool, help="whether to follow timestamp")
    parser.add_argument("--output-prefix", type=str, default="", help="Prefix for output files")
    parser.add_argument("--max-tokens", type=int, default=-1, help="Maximum number of tokens")
    parser.add_argument("--min-tokens", type=int, default=-1, help="Minimum number of tokens")
    args = parser.parse_args()

    if args.url is None:
        args.url = f"http://localhost:{args.port}/v1/chat/completions"
    
    if args.output_prefix is None or args.output_prefix == "":
        args.output_prefix = f"[{args.port}]_"
    if args.output_prefix and not args.output_prefix.endswith("_"):
        args.output_prefix += "_"
        
    if args.max_tokens > 0 and args.min_tokens < 0:
        args.min_tokens = 1 
    elif args.min_tokens > 0 and args.max_tokens < 0:
        args.max_tokens = 16384  # some large number

    asyncio.run(main(args))