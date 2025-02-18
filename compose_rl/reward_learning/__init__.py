# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from compose_rl.reward_learning.base_reward import (
    BaseReward,
    Reward,
    RewardModel,
)
from compose_rl.reward_learning.functional import (
    BadGenerationEndReward,
    IncreasingNumbersReward,
    OutputLengthReward,
    ShortResponseReward,
)
from compose_rl.reward_learning.hf_utils import (
    AutoModelForCausalLMWithRM,
    RewardModelConfig,
)
from compose_rl.reward_learning.inference_model import InferenceRewardModel
from compose_rl.reward_learning.model import (
    ComposerHFClassifierRewardModel,
    ComposerHFPairwiseRewardModel,
    ComposerMPTPairwiseRewardModel,
)

# Necessary to upload code when saving
RewardModelConfig.register_for_auto_class()
AutoModelForCausalLMWithRM.register_for_auto_class('AutoModel')

__all__ = [
    'BaseReward',
    'Reward',
    'RewardModel',
    'ComposerMPTPairwiseRewardModel',
    'ComposerHFPairwiseRewardModel',
    'ComposerHFClassifierRewardModel',
    'InferenceRewardModel',
    'BadGenerationEndReward',
    'IncreasingNumbersReward',
    'OutputLengthReward',
    'ShortResponseReward',
]
