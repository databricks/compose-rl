# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""MPT definition of an online RL Policy."""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from llmfoundry.models import MPTForCausalLM
from transformers import PreTrainedModel

from compose_rl.algorithms.online.model_methods import (
    CausalLMOutputWithPastAndValues,
    prepare_critic_values_for_training,
)
from compose_rl.algorithms.online.policy_configuration import MPTPolicyConfig

log = logging.getLogger(__name__)


class MPTPreTrainedPolicy(PreTrainedModel):
    config_class = MPTPolicyConfig
    base_model_prefix = 'model'
    _no_split_modules = ['MPTBlock']


class MPTForPolicy(MPTForCausalLM):

    def __init__(self, config: MPTPolicyConfig):
        super().__init__(config)

        self.dropout = nn.Dropout(self.config.critic_dropout)
        self.critic_head = nn.Linear(
            self.config.d_model,
            1,
            device=config.init_device,
        )
        self.critic_head._fsdp_wrap = True  # pyright: ignore

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

        encoding = super().forward(
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
