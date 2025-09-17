import random
import time
import logging
import asyncio


import aiohttp
import requests

from .http_utils import arequest_with_retry, get_default_connector
from .structs import InferenceEngineConfig, ModelRequest, ModelResponse, ParamSpec, WeightUpdateMeta

RID_CACHE_SIZE = 128

logger = logging.getLogger(__file__)


# TODO: have a base class for both vllm and sglang remote engines
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


    @property
    def num_servers(self) -> int:
        return len(self.addresses)

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
            "logprobs": gconfig.logprobs,
            "prompt_logprobs": gconfig.prompt_logprobs,
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

        output_tokens = result["results"][0]["outputs"][0]["token_ids"]
        output_logprobs = result["results"][0]["outputs"][0]["logprobs"]

        response = ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
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
        print(f"init_weight_update_group: payload={payload}")
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

    async def areset_prefix_cache(self):
        await asyncio.gather(
            *[
                arequest_with_retry(addr, endpoint="/reset_prefix_cache", method="POST", max_retries=1, timeout=self.config.request_timeout)
                for addr in self.addresses
            ]
        )
        return
