import random
import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
import uuid
import uvloop
import torch
import numpy as np

import aiohttp
import requests
from transformers import PreTrainedTokenizerFast

RID_CACHE_SIZE = 128

DEFAULT_RETRIES = 1
DEFAULT_REQUEST_TIMEOUT = 3600

logger = logging.getLogger(__file__)



def get_default_connector():
    return aiohttp.TCPConnector(limit=0, use_dns_cache=False, force_close=True)


async def arequest_with_retry(
    addr: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    session: aiohttp.ClientSession | None = None,
    method: str = "POST",
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    retry_delay: float = 1.0,
    verbose: bool = False,
) -> Dict:
    timeout = timeout or DEFAULT_REQUEST_TIMEOUT
    last_exception = None
    max_retries = max_retries or DEFAULT_RETRIES
    base_url = f"http://{addr}"
    url = f"{base_url}{endpoint}"

    timeo = aiohttp.ClientTimeout(
        total=timeout,
        sock_connect=timeout,
        connect=timeout,
    )
    if session is None:
        _session = aiohttp.ClientSession(
            timeout=timeo,
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
    else:
        _session = session

    for attempt in range(max_retries):
        try:
            if verbose:
                logger.info("enter client session, start sending requests")
            if method.upper() == "GET":
                ctx = _session.get(url, timeout=timeo)
            elif method.upper() == "POST":
                ctx = _session.post(url, json=payload, timeout=timeo)
            elif method.upper() == "PUT":
                ctx = _session.put(url, json=payload, timeout=timeo)
            elif method.upper() == "DELETE":
                ctx = _session.delete(url, timeout=timeo)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            async with ctx as response:
                if verbose:
                    logger.info("http requests return")
                response.raise_for_status()
                res = await response.json()
                if verbose:
                    logger.info("get http result")
                if session is None:
                    await _session.close()
                return res
        except (
            aiohttp.ClientError,
            aiohttp.ClientResponseError,
            asyncio.TimeoutError,
        ) as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            continue
    if session is None:
        await _session.close()
    raise RuntimeError(f"Failed after {max_retries} retries each. Payload: {payload}. Addr: {addr}. Endpoint: {endpoint}. Last error: {last_exception}")



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
    output_versions: list[int] = field(default_factory=list)
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
    nccl_param_specs: list[list[ParamSpec]] = field(default_factory=list)
    nccl_group_name: str = "update_weight_group"
    gen_tp_size: int = 1
    gen_world_size: int = 1


class RemoteSGLangEngine:

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
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemoteSGLangEngine does not support n_samples > 1. " +
                "Please call generate for multiple times with n_samples = 1."
            )
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop

        payload = {
            "input_ids": req.input_ids.copy(),
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []

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

        # Deal with rollout interruption
        # "abort" is the stop reason for later v0.4.9.post2 after
        # we call the pause_generation endpoint
        stop_reason = None
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):

            # loop until the generation is complete
            result = await arequest_with_retry(
                session=session,
                addr=server_addr,
                endpoint="/generate",
                payload=payload,
                method="POST",
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            meta_info = result["meta_info"]
            # Check if generation is complete
            finish_reason = meta_info["finish_reason"]
            stop_reason = finish_reason["type"]
            if (
                stop_reason == "abort"
                and finish_reason.get("message") == "Abort before prefill"
            ):
                continue

            # Parse response
            output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
            output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            # FIXME: Update with actual server versions
            accumulated_versions.extend([-1] * len(output_tokens))

            payload["input_ids"] += output_tokens
            sample_params["max_new_tokens"] -= len(output_tokens)

        if stop_reason == "abort":
            # If stop_reason is "abort", the only reason we exit the loop is
            # len(accumulated_output_tokens) >= gconfig.max_new_tokens
            # so the actual reason is length
            stop_reason = "length"
        await session.close()
        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
        )
        return response

    def update_weights(self, meta: WeightUpdateMeta):
        # for addr in self.addresses:
        #     res = requests.post(f"http://{addr}/pause_generation")
        #     res.raise_for_status()
        nccl_param_specs = [
            spec for param_specs in meta.nccl_param_specs for spec in param_specs
        ]

        async def _fn():
            tik = time.perf_counter()
            await asyncio.gather(
                *[
                    arequest_with_retry(
                        addr=addr,
                        endpoint="/update_weights_from_distributed",
                        payload={
                            "names": [pspec.name for pspec in nccl_param_specs],
                            "dtypes": [pspec.dtype for pspec in nccl_param_specs],
                            "shapes": [pspec.shape for pspec in nccl_param_specs],
                            "group_name": meta.nccl_group_name,
                        },
                        method="POST",
                        max_retries=1,
                        timeout=self.config.request_timeout,
                    )
                    for addr in self.addresses
                ]
            )

            logger.info(f"Distributed update weights done in {time.perf_counter() - tik}s")

        return uvloop.run(_fn())

    def init_weights_update_group(self, meta: WeightUpdateMeta):
        async def _fn():
            await asyncio.gather(
                *[
                    ainit_weights_update_group(addr, i, meta, self.config.request_timeout)
                    for i, addr in enumerate(self.addresses)
                ]
            )
        return uvloop.run(_fn())

async def ainit_weights_update_group(
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
        "backend": "nccl",
        "group_name": meta.nccl_group_name,
    }
    res = await arequest_with_retry(
        addr=addr,
        endpoint="/init_weights_update_group",
        payload=payload,
        method="POST",
        max_retries=1,
        timeout=request_timeout,
    )
    assert res["success"]
