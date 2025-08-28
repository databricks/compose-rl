import random
import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
import uuid
import torch
import numpy as np

import aiohttp
import requests
from transformers import PreTrainedTokenizerFast

from .http_utils import arequest_with_retry, get_default_connector

RID_CACHE_SIZE = 128

logger = logging.getLogger(__file__)



@dataclass
class InferenceEngineConfig:
    """Configuration for inference engine settings."""
    
    setup_timeout: float = 60.0  # Timeout in seconds for server setup/initialization
    request_timeout: float = 300.0  # Timeout in seconds for HTTP requests
    request_retries: int = 3  # Maximum number of retry attempts for requests


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for RL training."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    max_new_tokens: int = field(
        default=16384, metadata={"help": "Maximum number of tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0, metadata={"help": "Minimum number of tokens to generate."}
    )
    greedy: bool = field(
        default=False,
        metadata={"help": "Whether to use greedy decoding (max probability)."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability threshold (0.0, 1.0]."},
    )
    top_k: int = field(
        default=int(1e8),
        metadata={"help": "Number of highest probability tokens to consider."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature. Higher values increase diversity."},
    )
    stop_token_ids: list[int] = field(
        default_factory=list,
        metadata={"help": "Stop generation when encoutering these token ids."},
    )
    stop: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "One or multiple stop words. Generation will stop if one of these words is sampled."
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Penalizes tokens based on their frequency in generation so far. "
                "Must be between -2 and 2 where negative numbers encourage repeatment."
            )
        },
    )

@dataclass
class ModelRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: Optional[PreTrainedTokenizerFast] = None


@dataclass
class ModelResponse:
    # outputs
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    # what's this used for? tracking num of updates of the model?
    # output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "interrupt"] = "stop"
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: Optional[PreTrainedTokenizerFast] = None

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)


@dataclass
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def size(self) -> int:
        """Param bytes"""
        return getattr(torch, self.dtype).itemsize * np.prod(self.shape)


@dataclass
class WeightUpdateMeta:
    nccl_master_address: str = "127.0.0.1"
    nccl_master_port: int = 29500
    nccl_group_name: str = "update_weight_group"
    gen_tp_size: int = 1
    gen_world_size: int = 1


class RemoteVLLMEngine:

    def __init__(self, config: InferenceEngineConfig, addresses: list[str]):
        self.config = config

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []

        self.addresses = addresses

        if not self.addresses:
            raise RuntimeError("No configured SGLang servers.")

        self.server_idx = random.randint(0, len(self.addresses) - 1)


    def _wait_for_server(self, address: str):
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise RuntimeError("server launch failed")

    def check_health(self, base_url: str):
        # Check server endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=30)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def initialize(self):
        logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        logger.info("Servers are all ready!")

    def choose_server(self) -> str:
        server = self.addresses[self.server_idx]
        self.server_idx = (self.server_idx + 1) % len(self.addresses)
        return server

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Async version of generate using aiohttp."""
        # Prepare request payload
        gconfig = req.gconfig
        stop = gconfig.stop

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemoteVLLMEngine does not support n_samples > 1. " +
                "Please call generate multiple times with n_samples = 1."
            )
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "min_tokens": gconfig.min_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop

        payload = {
            "batched_prompt_token_ids": [req.input_ids.copy()],
            "sampling_params": sample_params,
        }

        # Make request
        start_time = time.perf_counter()

        # A single "rid" shares the same sever to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        # Create a new session because we don't know whether this method
        # is called in the workflow thread or the main thread.
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                sock_connect=self.config.request_timeout,
                connect=self.config.request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
        result = await arequest_with_retry(
            session=session,
            addr=server_addr,
            endpoint="/generate",
            payload=payload,
            method="POST",
            max_retries=self.config.request_retries,
            timeout=self.config.request_timeout,
        )
        await session.close()
        latency = time.perf_counter() - start_time

        accumulated_output_tokens = result["results"][0]["outputs"][0]["token_ids"]
        accumulated_output_logprobs = result["results"][0]["outputs"][0]["logprobs"]

        response = ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
        )
        return response

    async def aupdate_weight(self, param_spec: ParamSpec, empty_cache: bool = False):
        tik = time.perf_counter()
        await asyncio.gather(
            *[
                arequest_with_retry(
                    addr=addr,
                    endpoint="/update_weight",
                    payload={
                        "name": param_spec.name,
                        "dtype": param_spec.dtype,
                        "shape": param_spec.shape,
                        "empty_cache": empty_cache,
                    },
                    method="POST",
                    max_retries=1,
                    timeout=self.config.request_timeout,
                )
                for addr in self.addresses
            ]
        )

        logger.info(f"Distributed update weights done in {time.perf_counter() - tik}s")

        return

    @staticmethod
    async def _ainit_weight_update_group(
        addr: str,
        server_idx: int,
        meta: WeightUpdateMeta,
        request_timeout: float,
    ):
        rank_offset = 1 + server_idx * meta.gen_tp_size
        payload = {
            "master_address": meta.nccl_master_address,
            "master_port": str(meta.nccl_master_port),
            "rank_offset": rank_offset,
            "world_size": meta.gen_world_size + 1,
        }
        _ = await arequest_with_retry(
            addr=addr,
            endpoint="/init_weight_update_group",
            payload=payload,
            method="POST",
            max_retries=1,
            timeout=request_timeout,
        )
        # rely on HTTP 200 for success; server returns {"status": "ok"}

    async def ainit_weight_update_group(self, meta: WeightUpdateMeta):
        await asyncio.gather(
            *[
                self._ainit_weight_update_group(addr, i, meta, self.config.request_timeout)
                for i, addr in enumerate(self.addresses)
            ]
        )
        return 


