import argparse
from functools import cache
import json
from collections import defaultdict

#from benchmarks.multi_turn.bench_dataset import MessagesList, ConversationsMap
from vllm.inputs import (DataPrompt, PromptType, SingletonPrompt, TextPrompt,
                         TokensPrompt)
from vllm.config import get_current_vllm_config  

from typing import Any, Callable, Optional, Sequence, Union
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               init_tokenizer_from_configs)

ConvId = str

# A list of dicts (dicts with keys "role" and "content")
MessagesList = list[dict[str, str]]

# Map conversation ID to conversation messages
ConversationsMap = list[ConvId, MessagesList]

class RequestRecord:
    def __init__(self, req_id: int, parent_req_id: int, session_id: str, turn_id: int, timestamp: float, messages: MessagesList, response: str = ""):
        self.req_id = req_id
        self.parent_req_id = parent_req_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.timestamp = timestamp
        self.messages = messages  # List of message dicts up to this turn
        self.response = response
        self.ttft = 0.0  # Time to First Token
        self.cache_hit_rate = 0.0  # Placeholder for cache hit rate
        self.metrics: RequestOutput = None

class _FutureUsageMap:
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.last = None  # Keep for sequential chaining

    def __init__(self):
        self.usage_map: dict[tuple[int, int], tuple[int, int]] = {}
        self.turn_map: dict[int, int] = {}
            
        # tokenizer
        # tokenization_kwargs: dict[str, Any] = {}
        # self.tokenizer = init_tokenizer_from_configs(tokenization_kwargs)

        # block size
        vllm_config = get_current_vllm_config()
        self.block_size = vllm_config.cache_config.block_size

    def compute(
            self, 
            requests: list[dict[str, Any]], 
            tokenizer: AnyTokenizer
        ) -> dict[tuple[int, int], tuple[int, int]]:
        root = self.TrieNode()
        links = self.usage_map  # Now (prev_idx, curr_idx): max_overlap_len
        block_size = self.block_size
        
        for req_id, request in enumerate(requests, 1): # for every request
            node = root
            prev_req_id = None
            token_idx = 0

            # get first request of multi-turn conversation, 
            # use its sequence to build trie in case there is prefix caching  
            parent_req_id = request.parent_req_id if request.parent_req_id else None
            if request.parent_req_id is not None:
                while request.parent_req_id != -1:
                    request = requests[request.parent_req_id]
                self.turn_map[parent_req_id] = req_id
            
            # for every block that has prefix, 
            # create a map from previous (req_id, token_idx) to current request 
            prompt_token_ids = request.prompt_token_ids
            while token_idx < len(prompt_token_ids):
                # TODO: handle last block that is smaller than block_size
                block = tuple(prompt_token_ids[token_idx : token_idx + block_size]) # get tokens in a block
                if block not in node.children: # no prefix found
                    node.children[block] = self.TrieNode()
                child = node.children[block]
                if child.last is not None:
                    prev_req_id = child.last
                child.last = req_id
                node = child
                token_idx += block_size
                if prev_req_id:  # Only if prefix found
                    links[(prev_req_id, token_idx)] = (req_id, token_idx)
        return links

    def get_next_use(self, key: tuple[int, int]) -> tuple[int, int] | None:
        # prefix caching found
        if key in self.usage_map:
            return self.usage_map.get(key, None)
        # multi-turn conversation excluding prefix cached blocks
        elif key[0] in self.turn_map:
            return (self.turn_map[key[0]].get(key[1], None), key[1])
        return None

class TraceEntry:
    conv_id: str
    base_id: str
    prompt: str
    timestamp: float
    req_id: int
    parent_req_id: int
    
class FutureTurnMap:
    def __init__(self):
        self.turn_map: dict[int, int] = {}

    def compute(
            self, 
            requests: list[dict[str, Any]], 
        ) -> dict[int, int]:
        turn_map = self.turn_map  # Now req_id: {turn_id: next_turn_id}
        
        for req_id, request in enumerate(requests, 1): # for every request
            parent_req_id = request["parent_req_id"] if request["parent_req_id"] else None
            if parent_req_id is not None and parent_req_id != -1:
                turn_map[parent_req_id] = req_id

        return turn_map

    def compute(
            self, 
            requests: list[TraceEntry], 
        ) -> dict[int, int]:
        turn_map = self.turn_map  # Now req_id: {turn_id: next_turn_id}
        
        for req_id, request in enumerate(requests, 1): # for every request
            parent_req_id = request.parent_req_id if request.parent_req_id else None
            if parent_req_id is not None and parent_req_id != -1:
                turn_map[parent_req_id] = req_id

        return turn_map
    
    def get_next_use(self, key: tuple[int, int]) -> tuple[int, int] | None:
        if key is not None and key[0] in self.turn_map:
            return (self.turn_map.get(key[0], None), key[1])
        return None

    def save_turn_map_to_json(self, filepath: str):
        # Convert keys to strings for JSON compatibility
        turn_map_str_keys = {str(k): v for k, v in self.turn_map.items()}
        with open(filepath, 'w') as f:
            json.dump(turn_map_str_keys, f, indent=4)

    
FutureUsageMap = FutureTurnMap()