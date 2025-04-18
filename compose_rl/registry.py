# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Registry for all the components in the Compose RL."""

from compose_rl.ppo.kl_controller import (
    AdaptiveKLController,
    BallKLController,
    FixedKLController,
    KLPIDController,
)
from compose_rl.reward_learning import (
    BadGenerationEndReward,
    ComposerHFPairwiseRewardModel,
    ComposerMPTPairwiseRewardModel,
    GSM8KAnswerVeriferReward,
    GSM8KFormatVeriferReward,
    IncreasingNumbersReward,
    InferenceRewardModel,
    MATHVerifierReward,
    OutputLengthReward,
    ShortResponseReward,
)

RL_REWARD_REGISTRY = {
    'increasing_numbers': IncreasingNumbersReward,
    'output_length': OutputLengthReward,
    'short_response_reward': ShortResponseReward,
    'inference_reward_model': InferenceRewardModel,
    'mpt_pairwise': ComposerMPTPairwiseRewardModel,
    'hf_pairwise': ComposerHFPairwiseRewardModel,
    'bad_generation_end': BadGenerationEndReward,
    'gsm8k_answer_verifier': GSM8KAnswerVeriferReward,
    'gsm8k_format_verifier': GSM8KFormatVeriferReward,
    'math_verifier': MATHVerifierReward,
}

KL_CONTROLLER_REGISTRY = {
    'adaptive': AdaptiveKLController,
    'fixed': FixedKLController,
    'pid': KLPIDController,
    'ball': BallKLController,
}
