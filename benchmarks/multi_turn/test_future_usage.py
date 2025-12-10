from vllm.v1.core.sched.future_usage import FutureUsageMap
import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

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

entries = load_trace("100_trace.jsonl")
FutureUsageMap.compute(entries)
FutureUsageMap.save_turn_map_to_json("future_turn_map.json")
print(FutureUsageMap.get_next_use((1, 22)))