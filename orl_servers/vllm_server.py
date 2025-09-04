import argparse
import os
import signal
from typing import Any, Dict, List, Optional

import torch
import uvloop
from fastapi import Request
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, set_ulimit
import vllm.envs as envs
from vllm.sequence import Logprob
from vllm.outputs import RequestOutput

from .vllm_engine import AsyncEngine


def _to_torch_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.strip().lower()
    mapping: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "half": torch.float16,
        "int8": torch.int8,
        "i8": torch.int8,
        "int16": torch.int16,
        "i16": torch.int16,
        "int32": torch.int32,
        "i32": torch.int32,
        "int64": torch.int64,
        "i64": torch.int64,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[s]


def _extract_logprobs(logprob_list: list[dict[int, Logprob] | None], token_ids: list[int]) -> list[float]:
    """Extract logprob values from vLLM's list[dict[token_id, Logprob]] structure.
    
    Args:
        logprob_list: vLLM's logprobs structure (list[dict[token_id, Logprob]])
        token_ids: List of token IDs corresponding to the logprobs
        
    Returns:
        List of float logprob values, with 0.0 for missing entries
    """
    assert len(logprob_list) == len(token_ids), f"length mismatch: logprob_list: {len(logprob_list)}, token_ids: {len(token_ids)}"
    
    logprobs = []
    for token_id, logprob_dict in zip(token_ids, logprob_list):
        # Handle case where logprob_dict is None, e.g.,likely a special token like BOS/beginning-of-sequence
        if logprob_dict is None:
            logprobs.append(0.0)
            continue

        logprobs.append(logprob_dict[token_id].logprob)
    return logprobs


def _serialize_request_output(output: RequestOutput) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "request_id": output.request_id,
        "finished": output.finished,
        "outputs": [],
    }
    for o in output.outputs:
        token_ids = o.token_ids
        vllm_logprobs = o.logprobs
        if vllm_logprobs is not None:
            logprobs = _extract_logprobs(vllm_logprobs, token_ids)
        else:
            logprobs = None
        out["outputs"].append(
            {
                "token_ids": token_ids,
                "finish_reason": o.finish_reason,
                "stop_reason": o.stop_reason,
                "logprobs": logprobs,
            }
        )
    return out


class AsyncLLMServer:
    def __init__(self, args: argparse.Namespace):
        self.server_args = args

    async def run_server(self, **uvicorn_kwargs: Any) -> None:
        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)

        set_ulimit()

        def signal_handler(*_: Any) -> None:
            raise KeyboardInterrupt("terminated")

        signal.signal(signal.SIGTERM, signal_handler)

        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        engine_args = AsyncEngineArgs.from_cli_args(self.server_args)
        engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        app = build_app(self.server_args)

        # Initialize app state for default OpenAI-compatible routes
        vllm_config = await engine.get_vllm_config()
        await init_app_state(engine, vllm_config, app.state, self.server_args)

        # Wrap the existing engine with our AsyncLLM helper
        async_engine = AsyncEngine(engine)

        @app.post("/generate")
        async def _generate(request: Request):
            data = await request.json()
            batched_prompt_token_ids: List[List[int]] = data.get(
                "batched_prompt_token_ids", []
            )
            sampling_params_dict: Optional[Dict[str, Any]] = data.get(
                "sampling_params"
            )
            if sampling_params_dict is None:
                sampling_params = SamplingParams()
            else:
                sampling_params = SamplingParams(**sampling_params_dict)

            outputs = await async_engine.generate(
                batched_prompt_token_ids, sampling_params
            )
            return {"results": [_serialize_request_output(o) for o in outputs]}

        @app.post("/pause_generation")
        async def _pause_generation():
            await async_engine.pause_generation()
            return {"status": "ok"}

        @app.post("/continue_generation")
        async def _continue_generation():
            await async_engine.continue_generation()
            return {"status": "ok"}

        @app.post("/init_weight_update_group")
        async def _init_weight_update_group(request: Request):
            data = await request.json()
            master_addr: str = data.get("master_address")
            master_port = data.get("master_port")
            rank_offset: int = data.get("rank_offset")
            world_size: int = data.get("world_size")

            # Normalize types
            master_port_str = str(master_port)

            await async_engine.init_weight_update_group(
                master_addr, master_port_str, rank_offset, world_size
            )
            return {"status": "ok"}

        @app.post("/update_weight")
        async def _update_weight(request: Request):
            data = await request.json()
            name: str = data.get("name")
            dtype_str: str = data.get("dtype")
            shape: List[int] = data.get("shape")
            empty_cache: bool = data.get("empty_cache", False)

            dtype = _to_torch_dtype(dtype_str)
            await async_engine.update_weight(name, dtype, shape, empty_cache)
            return {"status": "ok"}

        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)

        shutdown_task = await serve_http(
            app,
            sock,
            host=self.server_args.host,
            port=self.server_args.port,
            log_level=self.server_args.uvicorn_log_level,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=self.server_args.ssl_keyfile,
            ssl_certfile=self.server_args.ssl_certfile,
            ssl_ca_certs=self.server_args.ssl_ca_certs,
            ssl_cert_reqs=self.server_args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

        await shutdown_task

        sock.close()

    def run_server_uvloop(self, **uvicorn_kwargs: Any) -> None:
        uvloop.run(self.run_server(**uvicorn_kwargs))


def main() -> None:
    parser: FlexibleArgumentParser = FlexibleArgumentParser(
        description="AsyncLLM REST API server (OpenAI-compatible + custom)."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    # Try to use ComposeRL's WorkerWrap if available, but don't hard-require it
    # Only set if not already provided by user via CLI/env
    if not getattr(args, "worker_extension_cls", None):
        args.worker_extension_cls = 'compose_rl.algorithms.online.generation_utils.vllm_utils.WorkerWrap'

    server = AsyncLLMServer(args)
    server.run_server_uvloop()


if __name__ == "__main__":
    main()


