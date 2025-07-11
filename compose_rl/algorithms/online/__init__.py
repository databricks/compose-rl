# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from compose_rl.algorithms.online.callback import OnPolicyCallback
from compose_rl.algorithms.online.single_controller_callback import SingleControllerOnPolicyCallback
from compose_rl.algorithms.online.kl_controller import (
    AdaptiveKLController,
    BallKLController,
    FixedKLController,
    KLPIDController,
)
from compose_rl.algorithms.online.model import (
    ComposerHFCriticFreePolicyLM,
    ComposerHFPolicyLM,
    ComposerMPTPolicyLM,
)
from compose_rl.algorithms.online.model_methods import \
    CausalLMOutputWithPastAndValues
from compose_rl.algorithms.online.policy_configuration import (
    HFPolicyConfig,
    MPTPolicyConfig,
)
from compose_rl.registry import kl_controllers

kl_controllers.register('adaptive', func=AdaptiveKLController)
kl_controllers.register('fixed', func=FixedKLController)
kl_controllers.register('pid', func=KLPIDController)
kl_controllers.register('ball', func=BallKLController)

__all__ = [
    'OnPolicyCallback',
    'SingleControllerOnPolicyCallback',
    'ComposerMPTPolicyLM',
    'ComposerHFPolicyLM',
    'ComposerHFCriticFreePolicyLM',
    'HFPolicyConfig',
    'MPTPolicyConfig',
    'CausalLMOutputWithPastAndValues',
]
