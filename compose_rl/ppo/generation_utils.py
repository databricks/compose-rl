# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""All generation utils for the llm or vllm engines."""

import logging
import time
from typing import Optional, Union

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


def generate(
    actor_critic: torch.nn.Module,
    vllm_engines: Optional[list],
    max_gen_len: int,
    batch: dict[str, torch.Tensor],
    pad_token_id: int,
    tokenizer: Tokenizer,
    generation_kwargs: dict,
) -> torch.Tensor:
    """Runs generate over the batch of prompts.

    Args:
        actor_critic (torch.nn.Module): The actor critic model to run generate over.
        vllm_engines (list): List of VLLM engines to use for generation.
        max_gen_len (int): Maximum generation length.
        batch (dict): The batch of data to run generate over.
        pad_token_id (int): The pad token id.
        tokenizer (Tokenizer): The tokenizer to use for decoding.
        generation_kwargs (dict): Generation keyword arguments.
    """
    cur_device = batch['prompt'].device
    prompt_tokens = batch['prompt']
    batch_size = batch['prompt'].shape[0]

    if vllm_engines is not None:
        prompt_all_gather_start_time = time.time()
        all_batched_prompts = dist.all_gather_object(prompt_tokens)
        log.info(
            f'took : {time.time() - prompt_all_gather_start_time} to gather prompts',
        )
        all_prompts = [
            prompt for batch in all_batched_prompts for prompt in batch
        ]

        batch_sizes = [len(batch) for batch in all_batched_prompts]

        if dist.get_global_rank() == 0:
            futs = []
            sampling_params = {
                'temperature': generation_kwargs.get('temperature', 1.0),
                'top_p': generation_kwargs.get('top_p', 1.0),
                'top_k': generation_kwargs.get('top_k', 50),
                'max_tokens': max_gen_len,
            }

            # We have to remove all pad tokens here
            all_prompts = [[
                token
                for token in prompt.detach().cpu().tolist()
                if token != pad_token_id
            ]
                           for prompt in all_prompts]

            # Calculate the base batch size
            batch_size = len(all_prompts) // len(vllm_engines)
            # Calculate the remainder (prompts that don't fit evenly)
            remainder = len(all_prompts) % len(vllm_engines)

            start_idx = 0
            for i, engine in enumerate(vllm_engines):
                # Assign one extra prompt to the first 'remainder' engines
                if i < remainder:
                    end_idx = start_idx + batch_size + 1
                else:
                    end_idx = start_idx + batch_size

                cur_prompts = all_prompts[start_idx:end_idx]
                cur_prompts = [
                    tokenizer.decode(prompt) for prompt in cur_prompts
                ]
                futs.append(
                    engine.generate.remote(
                        cur_prompts,
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
                all_responses.extend([
                    resp.outputs[0].token_ids for resp in result
                ])

            log.info(f'took: {time.time() - start_time} to gather futures')
            split_responses = []
            start = 0
            for size in batch_sizes:
                split_responses.append(all_responses[start:start + size])
                start += size
        else:
            split_responses = None

        dist.barrier()
        # scatter the respective responses to all other ranks
        local_responses = [None]
        start_time = time.time()
        torch.distributed.scatter_object_list(
            local_responses,
            split_responses,
            src=0,
        )
        log.info(f'took: {time.time() - start_time} to scatter prompts')

        local_responses = local_responses[0]

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
        padded_responses = torch.tensor(
            padded_responses,
            dtype=prompt_tokens.dtype,
            device=cur_device,
        )
        sequences = torch.cat([prompt_tokens, padded_responses], dim=-1)

    else:

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
        generated_dict = policy.generate( # type: ignore
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
