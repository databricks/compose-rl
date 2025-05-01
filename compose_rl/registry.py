# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.utils.registry_utils import create_registry

from compose_rl.interfaces.base_kl_controller import BaseKLController
from compose_rl.interfaces.base_reward import BaseReward

_rewards_description = (
    """The function rewards registry is used to register classes that implement function rewards.

    One example of this is IncreasingNumbersReward. See reward_learning/functional.py for examples.

    Args:
        tokenizer: Tokenizer: The tokenizer being used.
        kwargs: dict[Any, Any]: Additional keyword arguments.

    Returns:
        BaseReward: The reward class.
    """
)
rewards = create_registry(
    'llmfoundry',
    'rewards',
    generic_type=type[BaseReward],
    entry_points=True,
    description=_rewards_description,
)

_kl_controller_description = (
    """The KL Controller registry is used to register classes that implement KL Controller.

    One example of this is FixedKLController. See ppo/kl_controller.py for examples.

    Args:
        kl_config: dict[Any, Any]: The config for the kl controller class.
        device: Optional[torch.device]: The device to use for the kl controller.

    Returns:
        BaseKLController: The kl controller class.
    """
)
kl_controllers = create_registry(
    'llmfoundry',
    'kl_controllers',
    generic_type=type[BaseKLController],
    entry_points=True,
    description=_kl_controller_description,
)

__all__ = ['rewards', 'kl_controllers']
