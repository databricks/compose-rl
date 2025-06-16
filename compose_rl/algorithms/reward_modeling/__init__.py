# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from compose_rl.algorithms.reward_modeling.base_reward import (
    BaseReward,
    Reward,
    RewardModel,
)
from compose_rl.algorithms.reward_modeling.functional import (
    BadGenerationEndReward,
    GSM8KFormatVeriferReward,
    GSM8KVeriferReward,
    IncreasingNumbersReward,
    MATHFormatVerifierReward,
    MATHVerifierReward,
    OutputLengthReward,
    ShortResponseReward,
)
from compose_rl.algorithms.reward_modeling.hf_utils import (
    AutoModelForCausalLMWithRM,
    RewardModelConfig,
)
from compose_rl.algorithms.reward_modeling.inference_model import \
    InferenceRewardModel
from compose_rl.algorithms.reward_modeling.model import (
    ComposerHFClassifierRewardModel,
    ComposerHFCausalClassifierRewardModel,
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
rewards.register('gsm8k_verifier', func=GSM8KVeriferReward)
rewards.register('gsm8k_format_verifier', func=GSM8KFormatVeriferReward)
rewards.register('math_verifier', func=MATHVerifierReward)
rewards.register('math_format_verifier', func=MATHFormatVerifierReward)

__all__ = [
    'BaseReward',
    'Reward',
    'RewardModel',
    'ComposerMPTPairwiseRewardModel',
    'ComposerHFPairwiseRewardModel',
    'ComposerHFClassifierRewardModel',
    'ComposerHFCausalClassifierRewardModel',
    'InferenceRewardModel',
    'BadGenerationEndReward',
    'IncreasingNumbersReward',
    'OutputLengthReward',
    'ShortResponseReward',
    'GSM8KFormatVeriferReward',
    'GSM8KVeriferReward',
    'MATHVerifierReward',
    'MATHFormatVerifierReward',
]
