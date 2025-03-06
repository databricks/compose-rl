# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""DPO Composer Implementation."""

from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Union

import torch
from llmfoundry.models import ComposerHFCausalLM, ComposerMPTCausalLM
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from compose_rl.dpo.model_methods import DPOEnum, dpo_forward, dpo_loss

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

log = logging.getLogger(__name__)


class ComposerDPOLM(ComposerMPTCausalLM):
    """MPT model wrapper for DPO model."""

    def __init__(
        self,
        loss_type: str = 'dpo',
        beta: float = 0.1,
        label_smoothing: float = 0,
        sft_alpha: float = 0.0,
        **kwargs: Any,
    ):
        self.loss_type = DPOEnum(loss_type)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.sft_alpha = sft_alpha

        super().__init__(**kwargs)
        self.train_metrics = None  # DPOLM does not support eval_forward

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        return dpo_forward(
            model=self.model,
            tokenizer=self.tokenizer,
            batch=batch,
            policy_model_config=self.config,
            use_attention_sequence_id=self.model.transformer.
            attn_uses_sequence_id,
        )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: CausalLMOutputWithPast,
    ) -> None:
        raise ValueError('Eval forward is not implemented for ComposerDPOLM.')

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return dpo_loss(
            outputs,
            batch,
            self.loss_type,
            self.beta,
            self.label_smoothing,
            self.sft_alpha,
        )


class ComposerHFDPOLM(ComposerHFCausalLM):
    """HF class wrapper for DPO model."""

    def __init__(
        self,
        loss_type: str = 'dpo',
        beta: float = 0.1,
        label_smoothing: float = 0,
        sft_alpha: float = 0.0,
        **kwargs: Any,
    ):
        self.loss_type = DPOEnum(loss_type)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.sft_alpha = sft_alpha

        super().__init__(**kwargs)
        self.train_metrics = None  # DPOLM does not support eval_forward

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        return dpo_forward(
            model=self.model,
            tokenizer=self.tokenizer,
            batch=batch,
        )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: CausalLMOutputWithPast,
    ) -> None:
        raise ValueError('Eval forward is not implemented for ComposerHFDPOLM.')

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return dpo_loss(
            outputs,
            batch,
            self.loss_type,
            self.beta,
            self.label_smoothing,
            self.sft_alpha,
        )
