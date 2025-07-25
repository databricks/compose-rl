# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Online On-Policy RL callback."""

from __future__ import annotations

import logging
import time
from typing import Any, Union

import torch
from composer.core import (
    State,
    get_precision_context,
)
from composer.loggers import Logger
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from composer.utils import dist
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from compose_rl.algorithms.online.generation_utils import (
    broadcast_to_vllm,
    vllm_generate,
)
from compose_rl.algorithms.online.model import (
    ComposerHFPolicyLM,
    ComposerMPTPolicyLM,
)

# Import the base class
from compose_rl.algorithms.online.callback import OnPolicyCallback

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Policy = Union[ComposerHFPolicyLM, ComposerMPTPolicyLM]

__all__ = ['SingleControllerOnPolicyCallback']

log = logging.getLogger(__name__)


class SingleControllerOnPolicyCallback(OnPolicyCallback):
    """Callback for managing on-policy training in an RLHF loop.

    Ideally all the overwritten methods below should be implemented in the trainer actor instead of the callback, we kept them here for now to minimize a drastic refactor to PPO Callback code
    """

    def update_and_query_inference_engines(self, device: Any, vllm_engines: list[Any], model_update_group: dist.ProcessGroup):
        """Round trip to inference engines.
        
        Args:
            vllm_engines (list[Any]): The vllm engines to round trip to.
        """
        batch = device.batch_to_device(self._get_next_iter_prompts())
        self._update_inference_model(vllm_engines, model_update_group)
        self.batch_rollouts = self._interact_with_env(batch, vllm_engines)

    def iteration_start(self, state: State, logger: Logger):
        del logger  # unused

        self._get_reward(self.batch_rollouts)

        # Reset and initialize state train dataloader
        log.warning(
            'trainer._train_data_spec should be updated whenever the dataloader is updated',
        )
        # Train Dataloader
        state.set_dataloader(self.buffer, 'ep')
        state.train_dataloader = state.dataloader
        state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
            state.device_train_microbatch_size,
            state.auto_microbatching,
            state.train_dataloader,
        )

        # Update IFT KL
        self._update_ift_kl()

    def _interact_with_env(self, batch: dict[str, torch.Tensor], vllm_engines: list[Any]):
        """Have the policy interact with the environment.

        Here, we redo microbatching, and run generate appropriately. We add the environment
        interactions to the buffer.

        Args:
            batch (dict): the iteration level batch we want to interact with the environment.
        """
        max_gen_len = self.max_gen_len
        generation_kwargs = self.generation_kwargs
        with get_precision_context(self.precision), torch.no_grad():
            # If vllm engines are available, we use them to generate sequences in one go
            sequences = vllm_generate(
                vllm_engines=vllm_engines,
                batch=batch,
                max_gen_len=max_gen_len,
                generation_kwargs=generation_kwargs,
                tokenizer=self.tokenizer,  # type: ignore
                vllm_generate_function='generate',
            )
        # Add the prepared sequences to the batch again
        batch['sequences'] = sequences
        return batch


    def _update_inference_model(self, vllm_engines: list[Any], model_update_group: dist.ProcessGroup):
        start_time = time.time()
        log.info('Before broadcast to vLLM')
        broadcast_to_vllm(
            self.actor_critic,
            vllm_engines,
            model_update_group,
            device=torch.device('cuda'),
            loss_type=self.actor_critic.loss_type,  # type: ignore
        )
        log.info('Finished broadcasting to vLLM')
        log.info(f'Took: {time.time() - start_time} to broadcast to vllm.')
        dist.barrier()
