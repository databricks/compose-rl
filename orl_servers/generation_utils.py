import time
import asyncio
import logging
from typing import List

import torch

from .structs import GenerationHyperparameters, ModelRequest
from .vllm_remote import RemoteVLLMEngine
from .async_utils import run_async_sync


log = logging.getLogger(__name__)


async def _remote_vllm_generate_async(
    remote_engine: RemoteVLLMEngine,
    max_gen_len: int,
    generation_kwargs: dict,
    pad_token_id: int,
    all_prompts: list,
    batch_sizes: list,
) -> list:
    """Asynchronously generate completions using a RemoteVLLMEngine.

    Mirrors the return contract of the legacy `_vllm_generate` by returning a
    list of lists, where each inner list contains token-id sequences for the
    corresponding original minibatch shard (based on ``batch_sizes``).
    """
    # Normalize sampling params into GenerationHyperparameters
    top_k = generation_kwargs.get("top_k", -1)
    if top_k == -1:
        # Remote engine uses a very large number to effectively disable top-k
        top_k = int(1e8)

    gconfig_base = GenerationHyperparameters(
        n_samples=1,
        max_new_tokens=max_gen_len,
        min_new_tokens=generation_kwargs.get("min_new_tokens", 0),
        greedy=generation_kwargs.get("greedy", False),
        top_p=generation_kwargs.get("top_p", 1.0),
        top_k=top_k,
        temperature=generation_kwargs.get("temperature", 1.0),
        stop_token_ids=generation_kwargs.get("stop_token_ids", []),
        stop=generation_kwargs.get("stop"),
        frequency_penalty=generation_kwargs.get("frequency_penalty", 0.0),
        logprobs=generation_kwargs.get("logprobs", 1),
        prompt_logprobs=generation_kwargs.get("prompt_logprobs"),
    )

    # Remove pad tokens from all prompts
    cleaned_prompts: List[List[int]] = []
    for prompt in all_prompts:
        if isinstance(prompt, torch.Tensor):
            # TODO could speed this up with GPU by masking out pad tokens
            tokens = [
                token
                for token in prompt.detach().cpu().tolist()
                if token != pad_token_id
            ]
        else:
            tokens = [token for token in list(prompt) if token != pad_token_id]
        cleaned_prompts.append(tokens)

    # Dispatch one async generation per prompt
    tasks = []
    for prompt_ids in cleaned_prompts:
        req = ModelRequest(
            input_ids=prompt_ids,
            gconfig=gconfig_base,
        )
        tasks.append(asyncio.create_task(remote_engine.agenerate(req)))

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    log.info(f'took: {time.time() - start_time} to gather async generations')

    # Flatten responses in submission order (one output per input)
    all_responses: List[List[int]] = [resp.output_tokens for resp in results]

    # Distribute responses back to original device shards
    split_responses: List[list] = []
    start = 0
    for size in batch_sizes:
        split_responses.append(all_responses[start:start + size])
        start += size
    return split_responses


def _remote_vllm_generate(
    remote_engine: RemoteVLLMEngine,
    max_gen_len: int,
    generation_kwargs: dict,
    pad_token_id: int,
    all_prompts: list,
    batch_sizes: list,
) -> list:
    """Synchronous wrapper around `_remote_vllm_generate_async`.

    Keeps a similar call pattern as the original `_vllm_generate` while using
    the async ``agenerate`` under the hood for concurrency.
    """
    return run_async_sync(
        _remote_vllm_generate_async(
            remote_engine,
            max_gen_len,
            generation_kwargs,
            pad_token_id,
            all_prompts,
            batch_sizes,
        )
    )


