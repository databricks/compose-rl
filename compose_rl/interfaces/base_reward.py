# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokenizer = Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]

__all__ = ['BaseReward']


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
