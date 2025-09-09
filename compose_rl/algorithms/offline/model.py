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
from compose_rl.metrics.learning_metrics import TestLossMetric

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

log = logging.getLogger(__name__)


class ComposerMPTOfflinePolicyLM(ComposerMPTCausalLM):
    """MPT model wrapper for offline rl model."""

    def __init__(
        self,
        loss_type: str = 'apo',
        beta1: float = 0.5,
        beta2: float = 0.1,
        eta: float = 0.5, 
        multistep: bool = False,
        distributional_value_learning: bool = True,
        top_n_logits: int = 10,
        average_log_prob: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        self.loss_type = RegressionOfflineEnum(loss_type)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.distributional_value_learning = distributional_value_learning
        self.top_n_logits = top_n_logits
        self.multistep = multistep
        self.average_log_prob = average_log_prob
        self.temperature = temperature

        super().__init__(**kwargs)

        self.train_metrics = None  # DPOLM does not support eval_forward
        self.val_metrics = {metric.__class__.__name__ : metric for metric in [TestLossMetric()]}

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
            outputs = outputs,
            batch = batch,
            loss_type = self.loss_type,
            beta1 = self.beta1,
            beta2 = self.beta2,
            eta = self.eta, 
            multistep = self.multistep,
            distributional_value_learning = self.distributional_value_learning,
            top_n_logits = self.top_n_logits,
        )


class ComposerHFOfflinePolicyLM(ComposerHFCausalLM):
    """HF class wrapper for offline rl model."""

    def __init__(
        self,
        loss_type: str = 'apo',
        beta1: float = 0.5,
        beta2: float = 0.1,
        eta: float = 0.5, 
        multistep: bool = False,
        distributional_value_learning: bool = True,
        top_n_logits: int = 10,
        average_log_prob: bool = False,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        self.loss_type = RegressionOfflineEnum(loss_type)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.multistep = multistep
        self.distributional_value_learning = distributional_value_learning
        self.top_n_logits = top_n_logits
        self.average_log_prob = average_log_prob
        self.temperature = temperature

        super().__init__(**kwargs)
        self.train_metrics = None  # DPOLM does not support eval_forward
        print("initializing eval_metrics to be only TestLossMetric")
        self.val_metrics = {metric.__class__.__name__ : metric for metric in [TestLossMetric()]}


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
        outputs: CausalLMOutputWithPast | None = None,
    ) -> dict[str, torch.Tensor]:
        print("eval metrics: ", self.val_metrics)
        print("entered eval_forward")
        with torch.no_grad():
            fwd = self.forward(batch)
            loss = self.loss(fwd, batch)
        
        loss.update(fwd)    
        return loss


    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return offline_loss(
            outputs = outputs,
            batch = batch,
            loss_type = self.loss_type,
            beta1 = self.beta1,
            beta2 = self.beta2,
            eta = self.eta,
            multistep = self.multistep,
            distributional_value_learning = self.distributional_value_learning,
            top_n_logits = self.top_n_logits,
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
