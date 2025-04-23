# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from typing import Any, Optional, Union

from llmfoundry.models import MPTConfig
from transformers import AutoConfig, PretrainedConfig

from compose_rl.utils.consts import _MASTER_WEIGHTS_PRECISION


class MPTPolicyConfig(MPTConfig):
    model_type = 'mpt_policy'

    def __init__(
        self,
        joint_actor_critic: bool = True,
        critic_dropout: float = 0.0,
        value_clip_range: float = 0.2,
        value_loss_weight: float = 0.2,
        target_kl: float = 0.1,
        policy_clip_ratio: float = 0.15,
        compute_kl_loss: bool = True,
        **kwargs: Any,
    ):
        """Config Class for MPTPolicy.

        Args:
            joint_actor_critic (bool): Whether to use a joint actor-critic model.
            critic_dropout (float): Dropout rate for the critic model.
            value_clip_range (float): Clipping range for the value function.
            value_loss_weight (float): Weight of the value loss.
            target_kl (float): Target KL divergence for the PPO algorithm. If the KL with respect to the original
                generating policy, then we should early stop the current iteration of training.
            policy_clip_ratio (float): Clipping range for the policy.
            compute_kl_loss (bool): Whether to compute the KL divergence loss in the policy as a distillaiton loss.
                Otherwise we will compute it as an auxiliary reward.
            **kwargs (Any): Additional keyword arguments.
        """
        if not joint_actor_critic:
            raise ValueError(
                'We only support joint actor-critic right now. Please set joint_actor_critic=`True`.',
            )

        self.joint_actor_critic = joint_actor_critic
        self.critic_dropout = critic_dropout
        self.value_clip_range = value_clip_range
        self.value_loss_weight = value_loss_weight
        self.target_kl = target_kl
        self.policy_clip_ratio = policy_clip_ratio
        self.compute_kl_loss = compute_kl_loss

        super().__init__(**kwargs)


class HFPolicyConfig(PretrainedConfig):
    model_type = 'hf_policy'

    def __init__(
        self,
        base_model: Optional[Union[str, os.PathLike]
                            ] = 'meta-llama/Meta-Llama-3-70B-Instruct',
        base_config: Optional[PretrainedConfig] = None,
        pretrain_cfg: Optional[dict[str, Any]] = None,
        pretrained: bool = False,
        joint_actor_critic: bool = True,
        critic_dropout: float = 0.0,
        value_clip_range: float = 0.2,
        value_loss_weight: float = 0.2,
        target_kl: float = 0.1,
        policy_clip_ratio: float = 0.15,
        compute_kl_loss: bool = True,
        advantage_normalization: bool = True, # Default is true to fall back to GRPO
        length_normalize_policy_loss: bool = True, # Default is true to fall back to GRPO
        policy_clip_high_ratio: float | None = None,
        **kwargs: Any,
    ):
        """Config Class for HFPolicy."""
        if not joint_actor_critic:
            print(f"^^^^^^^^^^^^^^^^ In HFPolicyConfig init: {joint_actor_critic=}")
            print(f"Experimenting with joint_actor_critic=False")
            # raise ValueError(
            #     'We only support joint actor-critic right now. Please set joint_actor_critic=`True`.',
            # )
        super().__init__(**kwargs)

        self.base_model = base_model
        self.base_config = base_config if base_config is not None else AutoConfig.from_pretrained(
            base_model,
            torch_dtype=_MASTER_WEIGHTS_PRECISION,
        )

        temp_config = deepcopy(self.base_config)
        if not isinstance(temp_config, dict):
            temp_config = temp_config.__dict__
        for key, value in temp_config.items():
            if key not in ['_name_or_path', 'architectures']:
                setattr(self, key, value)

        self.pretrain_cfg = pretrain_cfg
        self.pretrained = pretrained

        self.joint_actor_critic = joint_actor_critic
        self.critic_dropout = critic_dropout
        self.value_clip_range = value_clip_range
        self.value_loss_weight = value_loss_weight
        self.target_kl = target_kl
        self.policy_clip_ratio = policy_clip_ratio
        self.compute_kl_loss = compute_kl_loss
        # Additional GRPO config overrides
        self.advantage_normalization = advantage_normalization
        self.length_normalize_policy_loss = length_normalize_policy_loss
        self.policy_clip_high_ratio = policy_clip_high_ratio
