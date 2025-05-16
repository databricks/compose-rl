# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from composer.utils import is_model_fsdp
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from compose_rl.ppo.modeling_utils import (
    CausalLMOutputWithPastAndValues,
    prepare_critic_values_for_training,
)
from compose_rl.ppo.policy_configuration import HFPolicyConfig
from compose_rl.utils.consts import _MASTER_WEIGHTS_PRECISION

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

log = logging.getLogger(__name__)


class AutoModelForCausalLMAsPolicy(PreTrainedModel):
    config_class = HFPolicyConfig

    # None of these are really true, but because this class inherits
    # from PreTrainedModel, we need to fake these to pass checks that transformers runs.
    # The real checks will be done when we call AutoModelForCausalLM in the constructor.
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config: PretrainedConfig, **kwargs: Any):
        super().__init__(config)
        self.config = config
        pretrain_cfg = config.pretrain_cfg
        pretrained = config.pretrained
        if pretrained:
            self.lm_backbone = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                config=config.base_config,
                **pretrain_cfg,
                torch_dtype=_MASTER_WEIGHTS_PRECISION,
            )
        else:
            # When downloading from hub, base config gets converted to dict
            # Redownload to make type PretrainedConfig
            if isinstance(config.base_config, dict):
                config.base_config = AutoConfig.from_pretrained(
                    config.base_model,
                    **config.base_config,
                    torch_dtype=_MASTER_WEIGHTS_PRECISION,
                )
            self.lm_backbone = AutoModelForCausalLM.from_config(
                config.base_config,
                **kwargs,
            )
        self.dropout = nn.Dropout(self.config.critic_dropout)
        self.critic_head = nn.Linear(
            self.config.hidden_size,
            1,
        )
        self.critic_head._fsdp_wrap = True  # pyright: ignore
        self.lm_backbone._fsdp_wrap = True

        self.critic_head._is_critic_head = True  # pyright: ignore

    def _init_weights(self, module: nn.Module) -> None:
        if hasattr(module, '_is_critic_head'):
            # Initialize weights with Xavier uniform
            nn.init.xavier_uniform_(module.weight)  # type: ignore
            # Initialize bias to zero
            if hasattr(module, 'bias'):
                nn.init.zeros_(module.bias)  # type: ignore
        else:
            super()._init_weights(module)

    def generate(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int,
        *args: Any,
        **kwargs: Any,
    ):
        if is_model_fsdp(self.lm_backbone):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # Note: We need to use the FSDP.summon_full_params context manager here because the generate function
            # does not seem to gather the weights for the LM head. This solution works because the tied weights of the LM head
            # are in the root FSDP module, and are summoned by the below context manager. See https://github.com/pytorch/pytorch/issues/100069
            # for more info.
            # Note: We use recurse=False here so that we only summon full params for the LM head, not the entire model.
            with FSDP.summon_full_params(
                self.lm_backbone,
                writeback=False,
                recurse=False,
            ):
                return self.lm_backbone.generate(
                    input_ids=input_ids,
                    pad_token_id=pad_token_id,
                    **kwargs,
                )

        else:
            return self.lm_backbone.generate(
                input_ids=input_ids,
                pad_token_id=pad_token_id,
                **kwargs,
            )

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        # Note need to update vocab size in base config as well so lm_head modification happens
        self.config.base_config.vocab_size = new_num_tokens
        model_embeds = super().resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        return model_embeds

    def set_input_embeddings(self, new_embeddings: Any):
        return self.lm_backbone.set_input_embeddings(new_embeddings)

    def get_input_embeddings(self):
        return self.lm_backbone.get_input_embeddings()

    def set_output_embeddings(self, new_embeddings: Any):
        return self.lm_backbone.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_backbone.get_output_embeddings()

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        **kwargs: Any,
    ) -> PreTrainedModel:
        config.pretrained = False
        model = cls(config, **kwargs)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args: Any,
        joint_actor_critic: bool = True,
        critic_dropout: float = 0.0,
        value_clip_range: float = 0.2,
        value_loss_weight: float = 0.2,
        target_kl: float = 0.1,
        policy_clip_ratio: float = 0.15,
        compute_kl_loss: bool = True,
        kl_estimator: Optional[str] = 'k1',
        kl_clip_range: Optional[float] = 40.0,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = 'main',
        use_safetensors: Optional[bool] = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        attn_implementation = kwargs.pop(
            'attn_implementation',
            'eager',
        )
        load_in_8bit = kwargs.pop('load_in_8bit', False)

        pretrained_model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            token=token,
            attn_implementation=attn_implementation,
            use_cache=False,
            torch_dtype=_MASTER_WEIGHTS_PRECISION,
        )

        if isinstance(pretrained_model_config, cls.config_class):
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config,
                cache_dir,
                ignore_mismatched_sizes,
                force_download,
                local_files_only,
                token,
                revision,
                use_safetensors,
                **kwargs,
            )

        pretrain_cfg = {
            'trust_remote_code': trust_remote_code,
            'token': token,
            'load_in_8bit': load_in_8bit,
            'attn_implementation': attn_implementation,
        }

        policy_config = HFPolicyConfig(
            base_model=pretrained_model_name_or_path,
            base_config=pretrained_model_config,
            hidden_size=pretrained_model_config.hidden_size,
            vocab_size=pretrained_model_config.vocab_size,
            pretrained=True,
            pretrain_cfg=pretrain_cfg,
            joint_actor_critic=joint_actor_critic,
            critic_dropout=critic_dropout,
            value_clip_range=value_clip_range,
            value_loss_weight=value_loss_weight,
            target_kl=target_kl,
            policy_clip_ratio=policy_clip_ratio,
            compute_kl_loss=compute_kl_loss,
            kl_estimator=kl_estimator,
            kl_clip_range=kl_clip_range,
        )

        model = cls(policy_config)

        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.ByteTensor] = None,
        prompt_len: Optional[torch.Tensor] = None,
        max_gen_len: Optional[int] = -1,
        action_mask: Optional[torch.Tensor] = None,
        zero_pad: bool = True,
        **kwargs: Any,
    ):
        encoding = self.lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        values = None
        if prompt_len is not None:
            assert max_gen_len is not None
            assert action_mask is not None
            dropout_enc = self.dropout(
                encoding.hidden_states[-1],  # pyright: ignore
            )

            all_values = self.critic_head(dropout_enc).squeeze(-1)
            values = prepare_critic_values_for_training(
                all_values,
                prompt_len,
                max_gen_len,
                action_mask,
                zero_pad,
            )

        ret_val = CausalLMOutputWithPastAndValues(
            loss=encoding.loss,
            logits=encoding.logits,
            past_key_values=encoding.past_key_values,
            attentions=encoding.attentions,
            hidden_states=encoding.hidden_states,
            values=values,
        )

        return ret_val
