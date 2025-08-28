import random
import time
import logging
import asyncio

import aiohttp
import requests

from .dataclasses import InferenceEngineConfig, ModelRequest, ModelResponse, ParamSpec, WeightUpdateMeta
from .http_utils import arequest_with_retry, get_default_connector

RID_CACHE_SIZE = 128

logger = logging.getLogger(__file__)




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

    async def aupdate_weights(self, param_specs: list[ParamSpec], group_name: str):
        # for addr in self.addresses:
        #     res = requests.post(f"http://{addr}/pause_generation")
        #     res.raise_for_status()

        tik = time.perf_counter()
        await asyncio.gather(
            *[
                arequest_with_retry(
                    addr=addr,
                    endpoint="/update_weights_from_distributed",
                    payload={
                        "names": [pspec.name for pspec in param_specs],
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": group_name,
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
    async def _ainit_weights_update_group(
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

    async def ainit_weights_update_group(self, meta: WeightUpdateMeta):
        await asyncio.gather(
            *[
                self._ainit_weights_update_group(addr, i, meta, self.config.request_timeout)
                for i, addr in enumerate(self.addresses)
            ]
        )
        return 


