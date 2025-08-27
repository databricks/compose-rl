#!/usr/bin/env python3
import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from transformers import AutoTokenizer


SERVER_HOST = "localhost"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_TOKENIZER: Optional[AutoTokenizer] = None


def _get_tokenizer() -> AutoTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOKENIZER = tok
    assert _TOKENIZER is not None
    return _TOKENIZER


def _encode_prompts(prompts: List[str]) -> List[List[int]]:
    tokenizer = _get_tokenizer()
    return [tokenizer.encode(p, return_tensors="pt").squeeze(0).tolist() for p in prompts]


def _decode_output_text(output_obj: Dict[str, Any]) -> str:
    try:
        outs = output_obj.get("outputs") or []
        if not outs:
            return ""
        token_ids = outs[0].get("token_ids") or []
        tok = _get_tokenizer()
        return tok.decode(token_ids, skip_special_tokens=True)
    except Exception:
        return ""


async def async_test_pause_continue_server() -> None:
    """Async pause/continue test using server HTTP API via aiohttp.

    Steps:
    1) Start a generate call
    2) Pause server (blocks new generation)
    3) Verify a new generate call is blocked (times out)
    4) Continue server and ensure blocked call completes
    5) Start a fresh generate after resume
    """
    initial_prompts = [
        "Write a detailed explanation of machine learning with examples and applications.",
        "Write a detailed explanation of Quantum mechanics with examples and applications.",
    ]
    blocked_prompts = ["Explain the history of artificial intelligence from its inception to modern times, including major milestones."]
    resumed_prompts = ["Describe the process of training a neural network step by step with mathematical details.",
                       "Describe the process of how a large language model works in details."]

    initial_encoded = _encode_prompts(initial_prompts)
    blocked_encoded = _encode_prompts(blocked_prompts)
    resumed_encoded = _encode_prompts(resumed_prompts)

    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
    }

    async def post_generate(session: aiohttp.ClientSession, batched: List[List[int]]):
        payload = {
            "batched_prompt_token_ids": batched,
            "sampling_params": sampling_params,
        }
        async with session.post(f"{BASE_URL}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            return await resp.json()

    async with aiohttp.ClientSession() as session:
        # 1) Start initial generation (don't await immediately)
        initial_task = asyncio.create_task(post_generate(session, initial_encoded))

        # Allow some time for the request to register
        await asyncio.sleep(0.5)

        # 2) Pause generation
        async with session.post(f"{BASE_URL}/pause_generation") as r:
            r.raise_for_status()

        # 3) Verify blocked new generation
        blocked_task = asyncio.create_task(post_generate(session, blocked_encoded))
        blocked_completed = True
        try:
            await asyncio.wait_for(asyncio.shield(blocked_task), timeout=1.0)
        except asyncio.TimeoutError:
            blocked_completed = False

        assert blocked_completed is False, "Blocked task should not complete while paused"

        # 4) Continue generation and wait for blocked task to complete
        async with session.post(f"{BASE_URL}/continue_generation") as r:
            r.raise_for_status()

        blocked_result = await blocked_task
        assert "outputs" in blocked_result and len(blocked_result["outputs"]) == 1
        blocked_text = _decode_output_text(blocked_result["outputs"][0])
        print(f"Blocked task completed after resume: {blocked_prompts[0]}\n{blocked_text}\n")

        # 5) Start new generation after resume
        resumed_result = await post_generate(session, resumed_encoded)
        assert "outputs" in resumed_result and len(resumed_result["outputs"]) == len(resumed_prompts)
        for i, out in enumerate(resumed_result["outputs"]):
            text = _decode_output_text(out)
            print(f"Resumed prompt {i+1}: {resumed_prompts[i]}\nResumed response {i+1}: {text}\n")

        # Await the initial task (it may have been cancelled server-side)
        try:
            initial_result = await initial_task
            for i, out in enumerate(initial_result.get("outputs", [])):
                text = _decode_output_text(out)
                print(f"Initial prompt {i+1}: {initial_prompts[i]}\nInitial response {i+1}: {text}\n")
        except asyncio.CancelledError:
            print("Initial task was cancelled during pause (expected)")


if __name__ == "__main__":
    asyncio.run(async_test_pause_continue_server())


