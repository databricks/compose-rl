# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

from compose_rl.utils.consts import _MASTER_WEIGHTS_PRECISION


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """Sequence Classification Output.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        scores (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class ValueHead(nn.Module):
    """Value head for the transformer which outputs n_labels values."""

    def __init__(self, n_labels: int, hidden_size: int, p_dropout: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(p_dropout)
        self.score = nn.Linear(hidden_size, n_labels, bias=False)
        torch.nn.init.normal_(
            self.score.weight,
            std=1 / np.sqrt(hidden_size + 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.score(hidden_states)
        return output


class RewardModelConfig(PretrainedConfig):
    model_type = 'pairwise_rm'

    def __init__(
        self,
        base_model: Optional[Union[str, os.PathLike]] = None,
        base_config: Optional[PretrainedConfig] = None,
        p_dropout: float = 0.0,
        n_labels: int = 1,
        bias: float = 0.0,
        return_logits: bool = False,
        pretrain_cfg: Optional[dict[str, Any]] = None,
        pretrained: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if base_model is None and base_config is None:
            raise ValueError(
                'base_model or base_config is required. Please set base_model to the path of the base model or base_config to the config of the base model.',
            )
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
        self.p_dropout = p_dropout
        self.n_labels = n_labels
        self.bias = bias
        self.return_logits = return_logits
        self.pretrain_cfg = pretrain_cfg if pretrain_cfg is not None else {}
        self.pretrained = pretrained


class AutoModelForCausalLMWithRM(PreTrainedModel):
    config_class = RewardModelConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: PretrainedConfig, **kwargs: Any):
        super().__init__(config)
        self.config = config
        pretrain_cfg = config.pretrain_cfg
        pretrained = config.pretrained

        print("#####################################")
        print(self.config.n_labels)
        print("#####################################")

        if pretrained:
            self.lm_backbone = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                config=config.base_config,
                **pretrain_cfg,
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
        self.value_head = ValueHead(
            n_labels=self.config.n_labels,
            hidden_size=self.config.hidden_size,
            p_dropout=self.config.p_dropout,
        )

    def generate(self, *args: Any, **kwargs: Any):
        return self.lm_backbone.generate(**kwargs)

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        output = self.lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )
        scores = self.value_head(
            output.hidden_states[-1],
        ).squeeze(-1) - self.config.bias

        logits = None
        if self.config.return_logits:
            logits = output.logits

        return SequenceClassifierOutput(
            loss=output.loss,
            scores=scores,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

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
        trust_remote_code = kwargs.pop('trust_remote_code', True)
        attn_implementation = kwargs.pop(
            'attn_implementation',
            'eager',
        )
        return_lm_logits = kwargs.pop('return_lm_logits', False)
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

        reward_model_config = RewardModelConfig(
            base_model=pretrained_model_name_or_path,
            base_config=pretrained_model_config,
            hidden_size=pretrained_model_config.hidden_size,
            return_logits=return_lm_logits,
            vocab_size=pretrained_model_config.vocab_size,
            pretrained=True,
            pretrain_cfg=pretrain_cfg,
        )

        model = cls(reward_model_config)

        return model
