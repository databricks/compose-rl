#!/usr/bin/env python3
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests
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


def _start_server() -> subprocess.Popen:
    cmd = [
        "orl-vllm-server",
        "--model",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--tensor-parallel-size",
        "1",
        "--trust-remote-code",
        "--max-model-len",
        "2048",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def _wait_for_server_ready(timeout_s: float = 120.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            # Probe a lightweight endpoint; pause_generation returns 200 when ready
            resp = requests.post(f"{BASE_URL}/pause_generation", timeout=2)
            if resp.status_code == 200:
                # Immediately continue generation to reset state
                requests.post(f"{BASE_URL}/continue_generation", timeout=2)
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("Server did not become ready in time")


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


def test_generate() -> None:
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum entanglement in simple terms.",
    ]
    batched = _encode_prompts(prompts)
    payload: Dict[str, Any] = {
        "batched_prompt_token_ids": batched,
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 64,
        },
    }
    resp = requests.post(f"{BASE_URL}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    assert "results" in data and isinstance(data["results"], list)
    assert len(data["results"]) == len(prompts)

    # Detokenize and print responses
    for i, out in enumerate(data["results"]):
        text = _decode_output_text(out)
        print(f"Prompt {i+1}: {prompts[i]}\nResponse {i+1}: {text}\n")


def test_pause_continue_blocks_and_resumes() -> None:
    # Pause generation to block new requests
    r = requests.post(f"{BASE_URL}/pause_generation", timeout=10)
    r.raise_for_status()

    prompts = ["What is the capital of France?"]
    batched = _encode_prompts(prompts)
    payload: Dict[str, Any] = {
        "batched_prompt_token_ids": batched,
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 32,
        },
    }

    blocked = False
    try:
        # Expect timeout while paused
        requests.post(f"{BASE_URL}/generate", json=payload, timeout=2)
    except requests.exceptions.Timeout:
        blocked = True

    assert blocked, "Generate should be blocked while paused"

    # Continue generation
    r = requests.post(f"{BASE_URL}/continue_generation", timeout=10)
    r.raise_for_status()

    # Now it should succeed
    print(f'prompt: {prompts[0]}')
    resp = requests.post(f"{BASE_URL}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    assert "results" in data and len(data["results"]) == 1
    text = _decode_output_text(data["results"][0])
    print(f"Resumed response: {text}")


def run():
    proc = _start_server()
    try:
        _wait_for_server_ready()
        test_generate()
        test_pause_continue_blocks_and_resumes()
        print("\n✅ async_llm_server basic tests passed (generate, pause/continue)")
    finally:
        # pass
        try:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            proc.kill()


if __name__ == "__main__":
    run()


