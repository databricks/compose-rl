# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from typing import MutableMapping, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from compose_rl.interfaces.base_reward import BaseReward

Tokenizer = Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]


class Reward(BaseReward):
    """Base class for generic rewards that do not wrap Composer models.

    This class (Reward) and its counterpart (RewardModel) serve as the two
    reward base classes used in this repo. They require their own initialization
    logic in the reward manager (see `../registry_builder.py`) and support
    different call signatures.

    Use this subclass for generic, functional rewards that do not use
    a Composer model. See `./functional.py` for examples of reward classes
    that inherit from this class.
    """

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class RewardModel(BaseReward):
    """Base class for rewards based on Composer models.

    This class (RewardModel) and its counterpart (Reward) serve as the two
    reward base classes used in this repo. They require their own initialization
    logic in the reward manager (see `../registry_builder.py`) and support
    different call signatures.

    Use this subclass for reward classes that are based on Composer
    models. See `./model.py` for examples of reward classes that inherit from
    this class.
    """

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.FloatTensor:
        raise NotImplementedError
