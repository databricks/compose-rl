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

