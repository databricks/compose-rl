# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Registry for all the components in the Compose RL."""

# Define empty registries first to avoid circular imports
RL_REWARD_REGISTRY = {}
KL_CONTROLLER_REGISTRY = {}
PGRM_FORMATTER_REGISTRY = {}

# Then import the components
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
    IncreasingNumbersReward,
    InferenceRewardModel,
    OutputLengthReward,
    PromptGuidedRewardModel,
    ShortResponseReward,
)
from compose_rl.reward_learning.pgrm_methods import (
    WeebSenseiPGRMFormatter,
)

# Populate registries after imports
RL_REWARD_REGISTRY.update({
    'increasing_numbers': IncreasingNumbersReward,
    'output_length': OutputLengthReward,
    'short_response_reward': ShortResponseReward,
    'inference_reward_model': InferenceRewardModel,
    'mpt_pairwise': ComposerMPTPairwiseRewardModel,
    'hf_pairwise': ComposerHFPairwiseRewardModel,
    'bad_generation_end': BadGenerationEndReward,
    'prompt_guided_reward_model': PromptGuidedRewardModel,
})

KL_CONTROLLER_REGISTRY.update({
    'adaptive': AdaptiveKLController,
    'fixed': FixedKLController,
    'pid': KLPIDController,
    'ball': BallKLController,
})

PGRM_FORMATTER_REGISTRY.update({
    'weeb_sensei': WeebSenseiPGRMFormatter,
})