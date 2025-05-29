# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""All generation utils for the llm or vllm engines."""

import gc
import logging
import time
from typing import Union

import ray
import torch
from composer.utils import dist
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from compose_rl.ppo.model import ComposerHFPolicyModel, ComposerMosaicPolicy
from compose_rl.utils import (
    flip_pad_token_usage_for_generate,
    flip_pad_token_usage_in_ffn,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Policy = Union[ComposerHFPolicyModel, ComposerMosaicPolicy]

log = logging.getLogger(__name__)


def hf_generate(
    actor_critic: torch.nn.Module,
    max_gen_len: int,
    batch: dict[str, torch.Tensor],
    pad_token_id: int,
    generation_kwargs: dict,
) -> torch.Tensor:
    """Runs hf generate over the batch of prompts.

    Args:
        actor_critic (torch.nn.Module): The actor critic model to run generate over.
        max_gen_len (int): Maximum generation length.
        batch (dict): The batch of data to run generate over.
        pad_token_id (int): The pad token id.
        generation_kwargs (dict): Generation keyword arguments.
    """
    cur_device = batch['prompt'].device
    prompt_tokens = batch['prompt']

    policy = actor_critic.model
    policy.eval()  # type: ignore
    # Adding a dummy forwards call.
    # We need this otherwise FSDP throws an error during a standard forward pass.
    policy( # type: ignore
        torch.tensor([[0]], dtype=torch.long, device=cur_device),
        attention_mask=torch.tensor([[1]],
                                    dtype=torch.bool,
                                    device=cur_device),
    )

    # Generate doesn't work if we unpad the FFN. So we need to check if we
    # need to flip the flag in the model.
    flipped_usage = flip_pad_token_usage_for_generate(
        policy,  # type: ignore
    )

    # We don't need to include EOS tokens since we mask out EOS tokens below
    generated_dict = actor_critic.generate( # type: ignore
        prompt_tokens,
        max_new_tokens=max_gen_len,
        return_dict_in_generate=True,
        synced_gpus=True,
        attention_mask=batch['prompt_attention_mask'],
        pad_token_id=pad_token_id,
        **generation_kwargs,
    )

    # We should flip the flag back after generate as needed.
    if flipped_usage:
        flip_pad_token_usage_in_ffn(policy)  # type: ignore

    # Sequences are [batch, seq_len + generated_len], covering the initial prompt and generated values
    sequences = generated_dict['sequences']  # type: ignore

    return sequences


def vllm_generate(
    vllm_engines: list,
    batch: dict,
    max_gen_len: int,
    generation_kwargs: dict,
    tokenizer: Tokenizer,
) -> torch.Tensor:
    """Run vllm generate on the prompts.

    Runs generate over a set of sequences in the batch. It also does extra computation
    that is required for later loss computation.

    Args:
        vllm_engines (list): List of vllm engines to run generate over.
        batch (dict): The batch of data to run generate over.
        max_gen_len (int): Maximum generation length.
        generation_kwargs (dict): Generation keyword arguments.
        tokenizer (Tokenizer): The actor critic's tokenizer.

    Returns:
        sequences (tensor): Tensor containing the prompt and generated sequences.
            The shape of the tensor is [batch_size, prompt_len + max_gen_len].
    """
    if type(vllm_engines) != list:
        raise TypeError(
            f'vllm_engines must be a list. Instead got {type(vllm_engines)=}',
        )
    # 1. Gather all prompts from all ranks
    # 2. Run generate over all prompts in one go
    # 3. Scatter the generated responses back to the correct rank
    # 4. Save the tokenized sequences in the batch for future use
    pad_token_id = tokenizer.pad_token_id  # type: ignore
    # Pull the necessary variables from the batch and self
    cur_device = batch['prompt'].device
    prompt_tokens = batch['prompt']

    prompt_all_gather_start_time = time.time()

    all_batched_prompts = dist.all_gather_object(prompt_tokens)
    batch_sizes = [len(batch) for batch in all_batched_prompts]

    log.info(
        f'took : {time.time() - prompt_all_gather_start_time} to gather prompts',
    )
    all_prompts = [prompt for batch in all_batched_prompts for prompt in batch]

    start_gen_time = time.time()
    if dist.get_global_rank() == 0:
        futs = []
        sampling_params = {
            'temperature': generation_kwargs.get('temperature', 1.0),
            'top_p': generation_kwargs.get('top_p', 1.0),
            'top_k': generation_kwargs.get('top_k', -1),
            'max_tokens': max_gen_len,
        }

        # We have to remove all pad tokens here
        all_prompts = [[
            token
            for token in prompt.detach().cpu().tolist()
            if token != pad_token_id
        ]
                       for prompt in all_prompts]

        # Generate with vllm
        # Calculate the base batch size
        vllm_batch_size = len(all_prompts) // len(vllm_engines)
        # Calculate the remainder (prompts that don't fit evenly)
        remainder = len(all_prompts) % len(vllm_engines)

        start_idx = 0
        for i, engine in enumerate(vllm_engines):
            # Assign one extra prompt to the first 'remainder' engines
            if i < remainder:
                end_idx = start_idx + vllm_batch_size + 1
            else:
                end_idx = start_idx + vllm_batch_size

            cur_prompts_ids = all_prompts[start_idx:end_idx]
            futs.append(
                engine.generate.remote(
                    prompt_token_ids=cur_prompts_ids,
                    use_token_ids=True,
                    sampling_params=sampling_params,
                ),
            )

            # Update the start index for the next iteration
            start_idx = end_idx

        start_time = time.time()
        results = ray.get(futs)
        all_responses = []

        # Get all of the ray futures
        for i, result in enumerate(results):
            # Each result is a list of responses this assumes one output per input
            all_responses.extend([resp.outputs[0].token_ids for resp in result])

        log.info(
            f'took: {time.time() - start_time} to gather futures',
        )

        # Distribute padded responses back to the correct device
        split_responses = []
        start = 0
        for size in batch_sizes:
            split_responses.append(
                all_responses[start:start + size],
            )
            start += size
    else:
        # Remove the memory from all gather as they are only used for the first rank
        all_batched_prompts = None
        all_prompts = None
        split_responses = None

    # Do another garbage collection and empty the cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dist.barrier()

    # Scatter the generated responses back to the correct rank
    local_responses = [None]
    start_time = time.time()
    torch.distributed.scatter_object_list(
        local_responses,
        split_responses,
        src=0,
    )
    local_responses = local_responses[0]

    log.info(f'took: {time.time() - start_time} to scatter prompts')

    # Pad the responses to the maximum length
    max_vllm_generated_len = max([
        len(response) for response in local_responses  # type: ignore
    ])
    padded_responses = []
    for sequence in local_responses:  # type: ignore
        sequence = list(sequence)
        if len(sequence) < max_vllm_generated_len:
            sequence = sequence + [
                pad_token_id,
            ] * (max_vllm_generated_len - len(sequence))

        padded_responses.append(sequence)

    # Convert the padded responses to a tensor
    padded_responses = torch.tensor(
        padded_responses,
        dtype=prompt_tokens.dtype,
        device=cur_device,
    )

    # Construct full sequences from the prompt and padded responses
    sequences = torch.cat([prompt_tokens, padded_responses], dim=-1)
    num_tokens_generated = sequences.size(1) - prompt_tokens.size(1)
    log.info(
        f'It took {time.time() - start_gen_time} to generate {num_tokens_generated} tokens',
    )
    return sequences
