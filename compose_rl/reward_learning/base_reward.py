# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, MutableMapping, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokenizer = Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]


class BaseReward(ABC):
    """Abstract base class for reward classes.

    Attributes:
        BLOCKING (bool): Flags whether the reward class
            should block execution when called (default, True)
            or can be run async (False). The reward manager
            will reference this attribute.

    Args:
        cfg: the config for the reward class.
        tokenizer: the tokenizer being used.
        **kwargs: additional keyword arguments.
    """

    # Whether the class blocks (True) or can be run async (False)
    BLOCKING: bool = True

    def __init__(
        self,
        cfg: dict[Any, Any],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer

    def validate_config(self) -> None:
        """Validates the config of the reward class.

        This method is called in the reward manager to ensure that the config is
        valid. It should be overridden in subclasses to perform any necessary
        validation.
        """
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor:
        """Method for calculating rewards.

        Exact signature will be specified in the subclasses that
        inherit from BaseReward.

        Returns:
            rewards (torch.FloatTensor): a tensor containing the [batch, seq]
                or the [batch, seq, n_labels] rewards
        """
        pass


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
