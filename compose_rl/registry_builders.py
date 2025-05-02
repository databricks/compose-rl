# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from typing import Any, Optional, Union

from llmfoundry.utils.registry_utils import construct_from_registry
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from compose_rl import registry
from compose_rl.ppo.kl_controller import BaseKLController
from compose_rl.reward_learning import BaseReward

__all__ = ['build_kl_controller', 'build_reward']

log = logging.getLogger(__name__)

Tokenizer = Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]


def build_kl_controller(
    name: str,
    kwargs: dict[str, Any],
    device: Optional[str] = None,
) -> BaseKLController:
    """Builds a load planner from the registry.

    Args:
        name (str): Name of the load planner to build.
        kl_config: dict[str, Any]: Additional keyword arguments.
        device: Optional[torch.device]: The device to use for the kl controller.

    Returns:
        BaseKLController: The kl controller.
    """
    if 'device' in kwargs:
        raise ValueError(
            f'`device` is a reserved keyword for kl controllers. Please remove it from the kwargs.',
        )

    _device = device if device is not None else 'cpu'
    kwargs['device'] = _device

    return construct_from_registry(
        name=name,
        registry=registry.kl_controllers,
        pre_validation_function=BaseKLController,
        post_validation_function=None,
        kwargs=kwargs,
    )


def build_reward(
    name: str,
    tokenizer: Tokenizer,
    kwargs: Optional[dict[str, Any]] = None,
) -> BaseReward:
    """Builds a reward model from the registry.

    Args:
        name (str): Name of the reward model to build.
        tokenizer: Tokenizer: The tokenizer to use for the reward model.
        kwargs: dict[str, Any]: Additional keyword arguments.

    Returns:
        BaseReward: The reward model.
    """
    if kwargs is None:
        kwargs = {}
    if 'tokenizer' in kwargs:
        raise ValueError(
            f'`tokenizer` is a reserved keyword for rewards. Please remove it from the kwargs.',
        )
    kwargs['tokenizer'] = copy.deepcopy(tokenizer)

    return construct_from_registry(
        name=name,
        registry=registry.rewards,
        pre_validation_function=BaseReward,
        post_validation_function=None,
        kwargs=kwargs,
    )
