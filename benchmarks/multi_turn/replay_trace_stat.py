#!/usr/bin/env python3
"""Replay trace requests produced by sharegpt_to_belady.py against a vLLM server."""

import argparse
import asyncio
import contextlib
import logging
from http import HTTPStatus
from bench_utils import TEXT_SEPARATOR, Color, logger
from statistics import mean
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, NamedTuple

import aiohttp

from vllm.v1.core.sched.future_usage import FutureUsageMap
from benchmark_serving_multi_turn import (
                ServerResponse, remove_prefix, 
                get_messages_token_count, get_token_count, 
                nanosec_to_millisec, RequestStats)
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
        
class RequestStatsExtended(NamedTuple):
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    start_time_ms: float
    input_num_turns: int
    input_num_tokens: int
    output_num_tokens: int
    output_num_chunks: int
    output_num_first_chunk_tokens: int
    approx_cached_percent: float
    conversation_id: str
    client_id: int
    num_cached_tokens: int
    input: Optional[str] = None
    output: Optional[str] = None
    reuse_ratio: Optional[float] = None
    provided_input_num_tokens: Optional[int] = None


class ServerResponse(NamedTuple):
    valid: bool
    ttft_ms: float  # time to first chunk
    tpot_ms: float  # time per output chunk (one or more tokens)
    latency_ms: float
    start_time_ms: float
    first_chunk: str  # first chunk of the content
    content: str  # includes the first_chunk
    num_chunks: int
    num_cached_tokens: int
    provided_num_input_tokens: int

    def __str__(self) -> str:
        return f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}"  # noqa: E501


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


def reset_cache(chat_url: str) -> None:
    url = chat_url.replace("/v1/chat/completions", "/reset_prefix_cache")
    try:
        requests.post(url, timeout=5)
        print(f"Reset prefix cache at {url}")
    except Exception as e:
        print(f"Warning: Failed to reset prefix cache at {url}: {e}")


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


async def send_request(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    chat_url: str,
    model: str,
    stream: bool = True,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    request_id: Optional[str] = None,
) -> ServerResponse:
    payload = {
        "model": model,
        "messages": messages,
        "seed": 0,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    if min_tokens is not None:
        payload["min_tokens"] = min_tokens

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = str(request_id)

    # Calculate the timeout for the request
    timeout_sec = 600
    if max_tokens is not None:
        # Assume TPOT of 200ms and use max_tokens to determine timeout
        timeout_sec = max(timeout_sec, int(max_tokens * 0.2))
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    valid_response = True
    ttft: Optional[float] = None
    chunk_delay: list[int] = []
    latency: Optional[float] = None
    first_chunk = ""
    generated_text = ""
    num_cached_tokens = 0
    provided_num_input_tokens: int = 0


    start_time: int = time.perf_counter_ns()
    most_recent_timestamp: int = start_time

    async with session.post(
        url=chat_url, json=payload, headers=headers, timeout=timeout
    ) as response:
        http_status = HTTPStatus(response.status)
        if http_status == HTTPStatus.OK:
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                if chunk == "[DONE]":
                    # End of stream
                    latency = time.perf_counter_ns() - start_time
                elif stream is False:
                    data = json.loads(chunk)
                    message = data["choices"][0]["message"]
                    assert message["role"] == "assistant"
                    generated_text += message["content"]
                    if usage := data.get("usage"):
                        num_cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                        provided_num_input_tokens = usage.get("prompt_tokens", 0)
                        #print(f"Cached tokens so far: {num_cached_tokens}")
                else:
                    timestamp: int = time.perf_counter_ns()
                    data = json.loads(chunk)

                    # Delta is the new content/text/data
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0]["delta"]
                        if delta.get("content", None):
                            if ttft is None:
                                # First token
                                first_token_time = time.perf_counter_ns()
                                ttft = first_token_time - start_time
                                first_chunk = delta["content"]
                            else:
                                # Decoding phase
                                chunk_delay.append(timestamp - most_recent_timestamp)

                            generated_text += delta["content"]
                    if usage := data.get("usage"):
                        num_cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                        provided_num_input_tokens = usage.get("prompt_tokens", 0)
                        #print(f"Cached tokens so far: {num_cached_tokens}")

                    most_recent_timestamp = timestamp
        else:
            valid_response = False
            content = await response.text()
            logger.warning(
                f"{Color.YELLOW}Received HTTP status {http_status.value} "
                f"({http_status.phrase}): {content}{Color.RESET}"
            )

    if latency is None:
        latency = -1.0
        if valid_response:
            # Streaming is disabled, latency was not set
            latency = time.perf_counter_ns() - start_time

    if ttft is None:
        # The response was a single chunk
        ttft = latency

    # Each chunk may include more than one token
    tpot: float = mean(chunk_delay) if len(chunk_delay) > 0 else 0.0
    num_chunks: int = len(chunk_delay)

    sr = ServerResponse(
        valid=valid_response,
        ttft_ms=nanosec_to_millisec(ttft) if ttft > 0.0 else -1.0,
        tpot_ms=nanosec_to_millisec(tpot),
        latency_ms=nanosec_to_millisec(latency),
        start_time_ms=nanosec_to_millisec(start_time),
        first_chunk=first_chunk,
        content=generated_text,
        num_chunks=num_chunks,
        num_cached_tokens=num_cached_tokens,
        provided_num_input_tokens=provided_num_input_tokens,
    )
    return sr


async def handle_entry(
    entry: TraceEntry,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    history: List[Dict[str, str]],
    stats: List[RequestStatsExtended],
    completion_events: Dict[int, asyncio.Event],
    start_time: float,
    tokenizer: AutoTokenizer,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> None:
    cm = semaphore if semaphore is not None else contextlib.nullcontext()
    async with cm:
        event = completion_events[entry.req_id]
        min_tokens = None if args.min_tokens < 0 else args.min_tokens
        max_tokens = None if args.max_tokens < 0 else args.max_tokens
        send_started = datetime.now()

        print(f"\r[{args.port}] sending req_id={entry.req_id} at {time.time() - start_time:.2f}s", end='', flush=True)

        response: ServerResponse = await send_request(
            session,
            messages,
            args.url,
            args.model,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            request_id=str(entry.req_id)
        )

    if response.valid is False:
        # Request failed
        event.set()
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

    rs = RequestStatsExtended(
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
        num_cached_tokens=response.num_cached_tokens,
        #input=entry.prompt,
        reuse_ratio=response.num_cached_tokens / input_num_tokens if input_num_tokens > 0 else 0.0,
        #output=response.content,
        provided_input_num_tokens=response.provided_num_input_tokens
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
    stats: List[RequestStatsExtended] = []
    
    FutureUsageMap.filename = f"/tmp/vllm_future_usage_map_{args.port}.pkl"
    FutureUsageMap.compute(entries)
    FutureUsageMap.write_to_file()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.follow_timestamp:
        semaphore = None
    else:
        semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        before_num_cache_queries, before_num_cache_hits = snapshot(args.url)
        for i, entry in enumerate(entries):
            completion_events.setdefault(entry.req_id, asyncio.Event())

            #scheduled_time = start_time + entry.timestamp
            #delay = scheduled_time - time.time()
            delay = entry.timestamp - entries[i - 1].timestamp if i > 0 else 0.0
            if delay > 0 and args.follow_timestamp:
                await asyncio.sleep(delay)
            await asyncio.sleep(0.5)

            if entry.parent_req_id != -1:
                parent_event = completion_events.get(entry.parent_req_id)
                if parent_event:
                    await parent_event.wait()

            history = histories.setdefault(entry.base_id, [])
            messages = list(history)
            messages.append({"role": "user", "content": entry.prompt})
            
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
            
            # Ensure output directory exists
            os.makedirs(args.output_dir, exist_ok=True)
            
            duration = time.time() - start_time

            total_tokens = raw_data["input_num_tokens"].sum() + raw_data["output_num_tokens"].sum()
            throughput = total_tokens / duration

            timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%m%d_%H%M")
            csv_filename = os.path.join(args.output_dir, f"replay_trace_stats_{timestamp}_{args.output_suffix}.csv")
            raw_data.to_csv(csv_filename, index=False)
            print(f"[{args.port}] Wrote replay stats for {len(raw_data)} requests to {csv_filename}", flush=True)
            
            after_num_cache_queries, after_num_cache_hits = snapshot(args.url)
            cache_queries = after_num_cache_queries - before_num_cache_queries
            cache_hits = after_num_cache_hits - before_num_cache_hits
            prefix_cache_hit_ratio = (cache_hits / cache_queries if cache_queries > 0 else 0.0)
            print(f"[{args.port}] num_queries: \33[34m{cache_queries}\33[0m, num_hits: \33[34m{cache_hits}\33[0m", flush=True)
            print(f"[{args.port}] Prefix cache hit ratio: \33[34m{prefix_cache_hit_ratio:.4f}\33[0m", flush=True)
            
            if args.output_suffix:
                instance = "lru" if "lru" in args.output_suffix else "belady"
            elif args.port == 40001:
                instance = "lru"
            elif args.port == 40002:
                instance = "belady"

            summary = { 
                "instance": instance,
                "prefix_cache_hit_ratio": prefix_cache_hit_ratio,
                "total_requests": len(stats),
                "max_concurrent_requests": args.max_concurrent_requests,
                "model": args.model,
                "timestamp": timestamp,
                "cache_hits": cache_hits,
                "cache_queries": cache_queries,
                "duration": f"{duration:.2f}s",
                "stats_file": os.path.basename(csv_filename),
                "trace_file": os.path.basename(args.trace_file),
                "throughput": throughput,
                "total_reuse_ratio": raw_data["num_cached_tokens"].sum() / raw_data["input_num_tokens"].sum() if raw_data["input_num_tokens"].sum() > 0 else 0.0,
            }
            json_filename = os.path.join(args.output_dir, f"summary_{timestamp}_{args.output_suffix}.json")
            with open(json_filename, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[{args.port}] Wrote summary stats to {json_filename}", flush=True)
            

def load_trace(trace_file: str) -> List[TraceEntry]:
    entries: List[TraceEntry] = []
    with open(trace_file, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if args.num_trace > 0 and i >= args.num_trace:
                break
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
        print(f"[{args.port}] Trace file is empty; nothing to replay.")
        return

    reset_cache(args.url)

    print(f"[{args.port}] Loaded {len(entries)} trace entries from {args.trace_file}")
    try:
        await replay_traces(entries, args)
    finally:
        FutureUsageMap.cleanup_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay trace against vLLM service")
    parser.add_argument("--trace-file", type=str, required=True, help="Path to the trace JSONL file")
    parser.add_argument( "--url", type=str, default=None, help="vLLM service URL")
    parser.add_argument("--port", type=int, default=40001, help="Port for the vLLM service")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--max-concurrent-requests", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--follow-timestamp", action='store_true', help="whether to follow timestamp")
    parser.add_argument("--output-suffix", type=str, default="", help="Prefix for output files")
    parser.add_argument("--output-dir", type=str, default="result", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=-1, help="Maximum number of tokens")
    parser.add_argument("--min-tokens", type=int, default=-1, help="Minimum number of tokens")
    parser.add_argument("--num-trace", type=int, default=300, help="number of trace entries to use")
    args = parser.parse_args()

    if args.url is None:
        args.url = f"http://localhost:{args.port}/v1/chat/completions"
    
    if args.output_suffix is None or args.output_suffix == "":
        args.output_suffix = f"[{args.port}]"
        
    if args.max_tokens > 0 and args.min_tokens < 0:
        args.min_tokens = 1 
    elif args.min_tokens > 0 and args.max_tokens < 0:
        args.max_tokens = 16384  # some large number

    asyncio.run(main(args))