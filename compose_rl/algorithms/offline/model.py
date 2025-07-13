# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Offline RL Composer Implementation."""

from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Union

import torch
from llmfoundry.models import ComposerHFCausalLM, ComposerMPTCausalLM
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from compose_rl.algorithms.offline.model_methods import (
    RegressionOfflineEnum,
    PairwiseOfflineEnum,
    offline_forward,
    offline_loss,
    pairwise_offline_forward,
    pairwise_offline_loss,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

log = logging.getLogger(__name__)


class ComposerMPTOfflinePolicyLM(ComposerMPTCausalLM):
    """MPT model wrapper for offline rl model."""

    def __init__(
        self,
        loss_type: str = 'apo',
        beta: float = 0.1,
        average_log_prob: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        self.loss_type = RegressionOfflineEnum(loss_type)
        self.beta = beta
        self.average_log_prob = average_log_prob
        self.temperature = temperature

        super().__init__(**kwargs)
        self.train_metrics = None  # DPOLM does not support eval_forward

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        return offline_forward(
            model=self.model,
            batch=batch,
            average_log_prob=self.average_log_prob,
            policy_model_config=self.config,
        )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: CausalLMOutputWithPast,
    ) -> None:
        raise ValueError('Eval forward is not implemented for ComposerDPOLM.')

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return offline_loss(
            outputs,
            batch,
            self.loss_type,
            self.beta,
        )


class ComposerHFOfflinePolicyLM(ComposerHFCausalLM):
    """HF class wrapper for offline rl model."""

    def __init__(
        self,
        loss_type: str = 'apo',
        beta: float = 0.1,
        average_log_prob: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        self.loss_type = RegressionOfflineEnum(loss_type)
        self.beta = beta
        self.average_log_prob = average_log_prob
        self.temperature = temperature

        super().__init__(**kwargs)
        self.train_metrics = None  # DPOLM does not support eval_forward

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        return offline_forward(
            model=self.model,
            batch=batch,
            average_log_prob=self.average_log_prob,
            temperature=self.temperature,
        )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: CausalLMOutputWithPast,
    ) -> None:
        raise ValueError('Eval forward is not implemented for ComposerHFDPOLM.')

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return offline_loss(
            outputs,
            batch,
            self.loss_type,
            self.beta,
        )


class ComposerMPTPairwiseOfflinePolicyLM(ComposerMPTCausalLM):
    """MPT model wrapper for DPO model."""

    def __init__(
        self,
        loss_type: str = 'dpo',
        beta: float = 0.1,
        label_smoothing: float = 0,
        sft_alpha: float = 0.0,
        average_log_prob: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        self.loss_type = PairwiseOfflineEnum(loss_type)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.sft_alpha = sft_alpha
        self.average_log_prob = average_log_prob
        self.temperature = temperature

        super().__init__(**kwargs)
        self.train_metrics = None  # DPOLM does not support eval_forward

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        return pairwise_offline_forward(
            model=self.model,
            tokenizer=self.tokenizer,
            batch=batch,
            average_log_prob=self.average_log_prob,
            policy_model_config=self.config,
            use_attention_sequence_id=self.model.transformer.
            attn_uses_sequence_id,  # type: ignore
            temperature=self.temperature,
        )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: CausalLMOutputWithPast,
    ) -> None:
        raise ValueError('Eval forward is not implemented for ComposerDPOLM.')

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return pairwise_offline_loss(
            outputs,
            batch,
            self.loss_type,
            self.beta,
            self.label_smoothing,
            self.sft_alpha,
        )


class ComposerHFPairwiseOfflinePolicyLM(ComposerHFCausalLM):
    """HF class wrapper for DPO model."""

    def __init__(
        self,
        loss_type: str = 'dpo',
        beta: float = 0.1,
        label_smoothing: float = 0,
        sft_alpha: float = 0.0,
        average_log_prob: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        self.loss_type = PairwiseOfflineEnum(loss_type)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.sft_alpha = sft_alpha
        self.average_log_prob = average_log_prob
        self.temperature = temperature

        super().__init__(**kwargs)
        self.train_metrics = None  # DPOLM does not support eval_forward

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        return pairwise_offline_forward(
            model=self.model,
            tokenizer=self.tokenizer,
            batch=batch,
            average_log_prob=self.average_log_prob,
            temperature=self.temperature,
        )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: CausalLMOutputWithPast,
    ) -> None:
        raise ValueError('Eval forward is not implemented for ComposerHFDPOLM.')

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return pairwise_offline_loss(
            outputs,
            batch,
            self.loss_type,
            self.beta,
            self.label_smoothing,
            self.sft_alpha,
        )
