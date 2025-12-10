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
from benchmark_serving_multi_turn import send_request, send_turn, ServerResponse, remove_prefix


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


class ServerResponse(NamedTuple):
    valid: bool
    ttft_ms: float  # time to first chunk
    tpot_ms: float  # time per output chunk (one or more tokens)
    latency_ms: float
    start_time_ms: float
    first_chunk: str  # first chunk of the content
    content: str  # includes the first_chunk
    num_chunks: int

    def __str__(self) -> str:
        return f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}"  # noqa: E501


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 2048,
    }
    headers = {"Content-Type": "application/json"}

    stream = payload.get("stream", False)
    valid_response = True
    ttft: Optional[float] = None
    chunk_delay: list[int] = []
    latency: Optional[float] = None
    first_chunk = ""
    generated_text = ""

    start_time: int = time.perf_counter_ns()
    most_recent_timestamp: int = start_time

    async with session.post(url, json=payload, headers=headers) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Request failed with status {response.status}: {await response.text()}"
            )
            
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
            else:
                timestamp: int = time.perf_counter_ns()
                data = json.loads(chunk)

                # Delta is the new content/text/data
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

                most_recent_timestamp = timestamp

        data = await response.json()
        try:
            return 
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Malformed response payload: {data}") from exc


async def handle_entry(
    entry: TraceEntry,
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: List[Dict[str, str]],
    history: List[Dict[str, str]],
    completion_events: Dict[int, asyncio.Event],
    start_time: float,
) -> None:
    event = completion_events[entry.req_id]
    send_started = datetime.datetime.now()
    print(f"sent req_id={entry.req_id} at {send_started}", flush=True)
    try:
        response = await send_request(session, url, model, messages)
    except Exception as exc:
        raise
    else:
        latency = time.time() - send_started
        history.extend(
            [
                {"role": "user", "content": entry.prompt},
                {"role": "assistant", "content": response},
            ]
        )
        elapsed = time.time() - start_time
    finally:
        event.set()


async def replay_traces(entries: List[TraceEntry], url: str, model: str) -> None:
    histories: Dict[str, List[Dict[str, str]]] = {}
    completion_events: Dict[int, asyncio.Event] = {}
    pending: List[asyncio.Task] = []
    
    FutureUsageMap.compute(entries)

    start_time = time.time()

    def print_event_status():
        print("Completion events status:", flush=True)
        for req_id, event in sorted(completion_events.items()):
            print(f"  req_id={req_id}: {'set' if event.is_set() else 'not set'}", flush=True)

    async with aiohttp.ClientSession() as session:
        for entry in entries:
            completion_events.setdefault(entry.req_id, asyncio.Event())

            scheduled_time = start_time + entry.timestamp
            delay = scheduled_time - time.time()
            if delay > 0:
                await asyncio.sleep(delay)

            parent_req_id = entry.parent_req_id
            if parent_req_id != -1:
                parent_event = completion_events.get(parent_req_id)
                if parent_event is None:
                    parent_event = asyncio.Event()
                    completion_events[parent_req_id] = parent_event
                if not parent_event.is_set():
                    await parent_event.wait()

            history = histories.setdefault(entry.base_id, [])
            messages = list(history)
            messages.append({"role": "user", "content": entry.prompt})

            print(f"sending req_id={entry.req_id} at {time.time() - start_time:.2f}s", flush=True)

            task = asyncio.create_task(
                handle_entry(
                    entry,
                    session,
                    url,
                    model,
                    messages,
                    history,
                    completion_events,
                    start_time,
                )
            )
            pending.append(task)

        if pending:
            await asyncio.gather(*pending)


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


async def main(trace_file: str, url: str, model: str) -> None:
    entries = load_trace(trace_file)
    if not entries:
        print("Trace file is empty; nothing to replay.")
        return

    print(f"Loaded {len(entries)} trace entries from {trace_file}")
    await replay_traces(entries, url, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay trace against vLLM service")
    parser.add_argument("--trace-file", type=str, required=True, help="Path to the trace JSONL file")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="vLLM service URL",
    )
    parser.add_argument("--port", type=int, default=40001, help="Port for the vLLM service")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    args = parser.parse_args()

    if args.url is None:
        args.url = f"http://localhost:{args.port}/v1/chat/completions"

    asyncio.run(main(args.trace_file, args.url, args.model))