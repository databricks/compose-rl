# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Online On-Policy RL callback."""

from __future__ import annotations

import logging
from typing import Union

from composer.core import State
from composer.loggers import Logger
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Import the base class
from compose_rl.algorithms.online.callback import OnPolicyCallback
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

    def iteration_start(self, state: State, logger: Logger):
        del logger  # unused

        print("[AT ITERATION START] batch_rollouts: ", self.batch_rollouts.keys())
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
