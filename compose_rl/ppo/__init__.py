# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from compose_rl.ppo.callback import PPOCallback
from compose_rl.ppo.kl_controller import (
    AdaptiveKLController,
    BallKLController,
    FixedKLController,
    KLPIDController,
)
from compose_rl.ppo.load_planner import PPOModelLoadPlanner
from compose_rl.ppo.model import (
    ComposerHFCriticFreePolicyModel,
    ComposerHFPolicyModel,
    ComposerMosaicPolicy,
)
from compose_rl.ppo.modeling_utils import CausalLMOutputWithPastAndValues
from compose_rl.ppo.policy_configuration import HFPolicyConfig, MPTPolicyConfig
from compose_rl.registry import kl_controllers

kl_controllers.register('adaptive', func=AdaptiveKLController)
kl_controllers.register('fixed', func=FixedKLController)
kl_controllers.register('pid', func=KLPIDController)
kl_controllers.register('ball', func=BallKLController)

__all__ = [
    'PPOCallback',
    'ComposerMosaicPolicy',
    'ComposerHFPolicyModel',
    'ComposerHFCriticFreePolicyModel',
    'HFPolicyConfig',
    'MPTPolicyConfig',
    'CausalLMOutputWithPastAndValues',
    'PPOModelLoadPlanner',
]
