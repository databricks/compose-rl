# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from compose_rl.reward_learning.base_reward import (
    BaseReward,
    Reward,
    RewardModel,
)
from compose_rl.reward_learning.functional import (  # MATHVerifierReward,
    BadGenerationEndReward,
    GSM8KAnswerVeriferReward,
    GSM8KFormatVeriferReward,
    IncreasingNumbersReward,
    MATHVerifierReward,
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

# Register rewards
from compose_rl.registry import rewards

rewards.register('increasing_numbers', func=IncreasingNumbersReward)
rewards.register('output_length', func=OutputLengthReward)
rewards.register('short_response_reward', func=ShortResponseReward)
rewards.register('inference_reward_model', func=InferenceRewardModel)
rewards.register('mpt_pairwise', func=ComposerMPTPairwiseRewardModel)
rewards.register('hf_pairwise', func=ComposerHFPairwiseRewardModel)
rewards.register('bad_generation_end', func=BadGenerationEndReward)
rewards.register('gsm8k_answer_verifier', func=GSM8KAnswerVeriferReward)
rewards.register('gsm8k_format_verifier', func=GSM8KFormatVeriferReward)
rewards.register('math_verifier', func=MATHVerifierReward)

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
    'GSM8KFormatVeriferReward',
    'GSM8KAnswerVeriferReward',
    'MATHVerifierReward',
]
