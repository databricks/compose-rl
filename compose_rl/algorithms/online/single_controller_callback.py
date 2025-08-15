# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Online On-Policy RL callback."""

from __future__ import annotations

import logging
from typing import Union
import torch

from composer.core import State, get_precision_context
from composer.core.data_spec import _default_split_batch
from composer.loggers import Logger
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Import the base class
from compose_rl.algorithms.online.callback import OnPolicyCallback, env_reward
from compose_rl.algorithms.online.model import (
    ComposerHFPolicyLM,
    ComposerMPTPolicyLM,
)
from compose_rl.utils import (
    add_right_padding,
    compute_advantages,
    dist_compute_masked_mean_and_var,
    get_decoded_sequence,
    get_entropies,
    get_log_probs,
    mask_eos,
    masked_mean,
    masked_sum,
    switch_left_to_right_padding,
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

    def iteration_end(self, state: State, logger: Logger):
        del logger  # unused
        # TODO: rethink logging
        # self._log_generations_to_logger(state)
