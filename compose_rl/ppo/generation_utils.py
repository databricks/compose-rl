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


def hf_generate(
    actor_critic: torch.nn.Module,
    max_gen_len: int,
    batch: dict[str, torch.Tensor],
    pad_token_id: int,
    generation_kwargs: dict,
) -> torch.Tensor:
    """Runs generate over the batch of prompts. Only for HF generate

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
