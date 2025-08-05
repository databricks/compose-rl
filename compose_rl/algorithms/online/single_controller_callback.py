# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Online On-Policy RL callback."""

from __future__ import annotations

import logging
from typing import Union
import torch

from composer.core import State
from composer.loggers import Logger
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Import the base class
from compose_rl.algorithms.online.callback import OnPolicyCallback, env_reward
from compose_rl.algorithms.online.model import (
    ComposerHFPolicyLM,
    ComposerMPTPolicyLM,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Policy = Union[ComposerHFPolicyLM, ComposerMPTPolicyLM]

__all__ = ['SingleControllerOnPolicyCallback']

log = logging.getLogger(__name__)


class SingleControllerOnPolicyCallback(OnPolicyCallback):
    """Callback for managing on-policy training in an RLHF loop.

    Ideally all the overwritten methods below should be implemented in the
    trainer actor instead of the callback, we kept them here for now to minimize
    a drastic refactor to PPO Callback code
    """

    def _get_reward(self, batch: dict[str, torch.Tensor]):
        """Compute rewards for a batch of generated sequences.

        Args:
            batch (dict): The batch containing generated sequences to compute rewards for.
        """
        all_rewards_dict = batch.pop('all_rewards_dict')

        env_outputs, prompts_and_gens, ref_outputs, empty_rewards_dict = env_reward(
            actor_critic=self.actor_critic,  # pyright: ignore
            reward_manager=self.reward_manager,
            batch=batch,
            max_gen_len=self.max_gen_len,
            precision=self.precision,
            device_train_microbatch_size=self.device_train_microbatch_size,
            tokenizer=self.tokenizer,  # type: ignore
            eos_token_ids=self.eos_token_ids,  # type: ignore
            kl_estimator=self.kl_estimator,
            kl_clip_range=self.kl_clip_range,
        )
        log.info(f"Empty rewards dict keys: {empty_rewards_dict.keys()}")

        self.prompts_and_gens.extend(prompts_and_gens)

        gen_batch_partial_outputs = (env_outputs, ref_outputs, all_rewards_dict)

        # For every partial output we want to resolve them together
        # And compute the global per iteration batch advantage's mean and variance
        resolved_outputs = self._resolve_outputs(
            batch,
            gen_batch_partial_outputs,
        )

        # Delete Non-tensor keys for training batch
        for key in ['verified_answer', 'messages']:
            if key in resolved_outputs.keys():
                del resolved_outputs[key]

        # We need to split the resolved outputs into minibatches
        for idx in range(
            batch['prompt_id'].shape[0] // self.device_train_batch_size,
        ):
            minibatch = self._extract_minibatch(
                resolved_outputs,
                idx,
                self.device_train_batch_size,
            )
            self.buffer.add(minibatch)

        # Making sure we correctly parsed the minibatches
        assert len(
            self.buffer,
        ) == self.num_batches_per_update, f'{len(self.buffer)} != {self.num_batches_per_update}'

        self.actor_critic.train()

    def iteration_start(self, state: State, logger: Logger):
        del logger  # unused

        self._get_reward(self.batch_rollouts)  # type: ignore

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

    def iteration_end(self, state: State, logger: Logger):
        del logger  # unused
        self._log_generations_to_logger(state)
        self._increment_rl_iter()
        self.buffer.reset()
