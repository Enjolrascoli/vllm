import os
import asyncio
import json
import random
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from bench_dataset import (
    conversations_list_to_dict,
    ConversationsMap,
    MessagesList,
)
from vllm.v1.core.sched.future_usage import FutureUsageMap

# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # Example model; adjust as needed
SHAREGPT_FILE = "sharegpt_conv_128.json"  # Path to ShareGPT JSON file
NUM_SESSIONS = 4  # Limit to number of conversations for testing; adjust as needed
POISSON_LAMBDA = 1.0  # Lambda for Poisson distribution (requests per second)
MAX_TURNS_PER_CONV = 10  # Max turns to process per conversation
SEED = 42

# Data structures
class RequestRecord:
    def __init__(
        self,
        req_id: int,
        parent_req_id: int,
        session_id: str,
        turn_id: int,
        timestamp: float,
        messages: MessagesList,
        response: str = "",
        prompt_token_ids: List[int] = []
    ):
        self.req_id = req_id
        self.parent_req_id = parent_req_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.timestamp = timestamp
        self.messages = messages  # List of message dicts up to this turn
        self.response = response
        self.prompt_token_ids = prompt_token_ids
        self.ttft = 0.0  # Time to First Token
        self.cache_hit_rate = 0.0  # Placeholder for cache hit rate
        self.metrics: RequestOutput = None

    def to_dict(self):
        return {
            "req_id": self.req_id,
            "parent_req_id": self.parent_req_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "messages": self.messages,
            "response": self.response,
            "ttft": self.ttft,
            "cache_hit_rate": self.cache_hit_rate,
            "prompt_token_ids": self.prompt_token_ids,
        }

def load_sharegpt_conversations(file_path: str, num_sessions: int) -> ConversationsMap:
    """Load and limit ShareGPT conversations to a ConversationsMap."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Convert to ConversationsMap (dict of conv_id to MessagesList)
    conversations = conversations_list_to_dict(data)
    # Limit to num_sessions
    limited_convs = {k: v[:MAX_TURNS_PER_CONV] for i, (k, v) in enumerate(conversations.items()) if i < num_sessions}
    return limited_convs

async def process_conversation(
        llm: LLM,
        session_id: str,
        messages: MessagesList,
        poisson_lambda: float,
        sampling_params: SamplingParams,
        traces: List[RequestRecord],
        start_time: float
    ):
    """Process a single conversation with Poisson start delay."""
    # Poisson delay for conversation start
    delay = random.expovariate(poisson_lambda)
    await asyncio.sleep(delay)

    history = []
    for turn_id in range(0, len(messages), 2):  # Process user-assistant pairs
        user_msg = messages[turn_id]
        history.append(user_msg)

        timestamp = time.time() - start_time

        result = llm.chat(history, sampling_params)
        response = result[0].outputs[0].text if result else ""
        req_id = result[0].request_id if result else 0

        # Find parent_req_id
        if turn_id == 0:
            parent_req_id = -1
        else:
            prev_turn_id = (turn_id // 2) - 1
            prev_trace = next((t for t in traces if t.session_id == session_id and t.turn_id == prev_turn_id), None)
            parent_req_id = prev_trace.req_id if prev_trace else -1

        # Record
        trace = RequestRecord(req_id, parent_req_id, session_id, turn_id // 2, timestamp, history.copy(), response, prompt_token_ids=result[0].prompt_token_ids)
        # Placeholder: Compute cache hit rate from vLLM stats (e.g., parse profiling output)
        # Example: trace.cache_hit_rate = llm.get_cache_hit_rate()  # Hypothetical API
        #trace.ttft = result[0].metrics.time_to_first_token if result else 0.0
        trace.metrics = result[0].metrics if result else None
        traces.append(trace)

        # Add assistant response for next turn
        if turn_id + 1 < len(messages):
            history.append({"role": "assistant", "content": response})

async def replay_turn(
        llm: LLM,
        sampling_params: SamplingParams,
        trace: RequestRecord,
        start_time: float
    ) -> RequestRecord:
    await asyncio.sleep(trace.timestamp)
    timestamp = time.time() - start_time

    result = llm.chat(trace.messages, sampling_params)

    req_id = result[0].request_id if result else 0
    new_trace = RequestRecord(
        req_id,
        trace.parent_req_id,
        trace.session_id,
        trace.turn_id,
        timestamp,
        trace.messages,
        result[0].outputs[0].text if result else "" , 
        prompt_token_ids=result[0].prompt_token_ids
    )
    #new_trace.ttft = result[0].metrics.time_to_first_token if result else 0.0
    new_trace.metrics = result[0].metrics if result else None
    return new_trace

async def first_pass(llm: LLM, conversations: ConversationsMap, poisson_lambda: float) -> List[RequestRecord]:
    """First pass: Send requests with conversation starts following Poisson distribution, record timestamps and responses."""
    traces = []
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)  # Adjust as needed
    start_time = time.time()

    tasks = []
    for session_id, messages in conversations.items():
        task = asyncio.create_task(process_conversation(llm, session_id, messages, poisson_lambda, sampling_params, traces, start_time))
        tasks.append(task)

    await asyncio.gather(*tasks)
    return traces

async def second_pass(llm: LLM, traces: List[RequestRecord]) -> List[RequestRecord]:
    """Second pass: Replay requests at recorded timestamps, measure metrics."""
    FutureUsageMap.compute(traces, llm.get_tokenizer())
    updated_traces = []
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    # Sort traces by timestamp for optimal replay (Belady's Min simulation)
    traces.sort(key=lambda r: r.timestamp)

    start_time = time.time()
    tasks = []
    for trace in traces:
        task = asyncio.create_task(replay_turn(llm, sampling_params, trace, start_time))
        tasks.append(task)

    updated_traces = await asyncio.gather(*tasks)
    return updated_traces

async def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

    # Load ShareGPT conversations
    conversations = load_sharegpt_conversations(SHAREGPT_FILE, NUM_SESSIONS)
    print(f"Loaded {len(conversations)} conversations from {SHAREGPT_FILE}")

    # Initialize LLM
    llm = LLM(model=MODEL_NAME, 
              seed=SEED, 
              gpu_memory_utilization=0.8,
              disable_log_stats=False,
              #max_model_len=4096, 
              #enforce_eager=True
              )  # Adjust args as needed

    # First pass
    print("Starting first pass...")
    first_pass_traces = await first_pass(llm, conversations, POISSON_LAMBDA)
    with open("first_pass_traces.json", "w") as f:
        json.dump([t.__dict__ for t in first_pass_traces], f, indent=2)
    print(f"Saved {len(first_pass_traces)} first pass traces to first_pass_traces.json")

    # Second pass
    print("Starting second pass...")
    second_pass_traces = await second_pass(llm, first_pass_traces)
    with open("second_pass_traces.json", "w") as f:
        json.dump([t.__dict__ for t in second_pass_traces], f, indent=2)
    print(f"Saved {len(second_pass_traces)} second pass traces to second_pass_traces.json")
    
    metrics = llm.get_metrics()
    metrics_dicts = [metric.__dict__ for metric in metrics]
    with open('metrics.json', 'w') as f:
        json.dump(metrics_dicts, f, indent=4)
    print(f"Saved metrics to metrics.json")

if __name__ == "__main__":
    asyncio.run(main())