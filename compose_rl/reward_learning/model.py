# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Reward Model Composer Implementation."""

import logging
from typing import Any, Mapping, MutableMapping, Optional

import torch
from llmfoundry.models import ComposerMPTCausalLM

from compose_rl.reward_learning.base_reward import RewardModel, Tokenizer
from compose_rl.reward_learning.hf_utils import SequenceClassifierOutput
from compose_rl.reward_learning.model_methods import (
    ClassifierRewardEnum,
    PairwiseRewardEnum,
    classifier_forward,
    classifier_loss,
    pairwise_forward,
    pairwise_loss,
)
from compose_rl.reward_learning.modeling_hf import \
    ComposerHFSequenceClassification
from compose_rl.reward_learning.modeling_mpt import MPTForSequenceClassification

log = logging.getLogger(__name__)


class ComposerHFPairwiseRewardModel(
    ComposerHFSequenceClassification,
    RewardModel,
):

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        loss_type: str = 'bt',
        return_lm_logits: bool = False,
        return_last: bool = True,
        **kwargs: Any,
    ):
        self.loss_type = PairwiseRewardEnum(loss_type)
        self.return_lm_logits = return_lm_logits
        self.return_last = return_last

        config_overrides = {
            'return_logits': return_lm_logits,
        }

        if 'config_overrides' in kwargs:
            config_overrides.update(kwargs.pop('config_overrides'))

        self.min_threshold = kwargs.pop('min_threshold', None)
        self.max_threshold = kwargs.pop('max_threshold', None)

        super().__init__(
            tokenizer=tokenizer,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            config_overrides=config_overrides,
            **kwargs,
        )

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        is_inference = batch.get('is_inference', False)
        if is_inference:
            return self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_lm_logits=self.return_lm_logits,
            ).scores
        else:
            return pairwise_forward(
                model=self.model,
                tokenizer=self.tokenizer,
                batch=batch,
                return_last=self.return_last,
                return_lm_logits=self.return_lm_logits,
            )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: Optional[SequenceClassifierOutput] = None,
    ) -> dict[str, torch.Tensor]:
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs: SequenceClassifierOutput,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return pairwise_loss(
            outputs,
            batch,
            self.loss_type,
        )


class ComposerHFClassifierRewardModel(
    ComposerHFSequenceClassification,
    RewardModel,
):

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        loss_type: str = 'bce',
        return_lm_logits: bool = False,
        return_last: bool = True,
        **kwargs: Any,
    ):
        self.loss_type = ClassifierRewardEnum(loss_type)
        self.return_lm_logits = return_lm_logits
        self.return_last = return_last

        config_overrides = {
            'return_logits': return_lm_logits,
        }

        if 'config_overrides' in kwargs:
            config_overrides.update(kwargs.pop('config_overrides'))

        self.min_threshold = kwargs.pop('min_threshold', None)
        self.max_threshold = kwargs.pop('max_threshold', None)

        super().__init__(
            tokenizer=tokenizer,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            config_overrides=config_overrides,
            **kwargs,
        )

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        ret_val = classifier_forward(
            model=self.model,
            tokenizer=self.tokenizer,
            batch=batch,
            return_last=self.return_last,
            return_lm_logits=self.return_lm_logits,
        )

        return ret_val

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: Optional[SequenceClassifierOutput] = None,
    ) -> dict[str, torch.Tensor]:
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs: SequenceClassifierOutput,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return classifier_loss(
            outputs,
            batch,
            self.loss_type,
        )


class ComposerMPTPairwiseRewardModel(ComposerMPTCausalLM, RewardModel):
    """MPT model wrapper for Pairwise/BT reward model."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[list] = None,
        loss_type: str = 'bt',
        return_lm_logits: bool = False,
        return_last: bool = True,
        **kwargs: Any,
    ):
        self.loss_type = PairwiseRewardEnum(loss_type)
        self.return_lm_logits = return_lm_logits
        self.return_last = return_last

        kwargs[
            'loss_fn'
        ] = 'torch_crossentropy'  # NOTE: passing in dummy value to overwrite
        super().__init__(
            tokenizer=tokenizer,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            **kwargs,
        )

    @property
    def model_class(self) -> type[MPTForSequenceClassification]:
        return MPTForSequenceClassification

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        is_inference = batch.get('is_inference', False)
        if is_inference:
            return self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_lm_logits=self.return_lm_logits,
            ).scores
        else:
            return pairwise_forward(
                model=self.model,
                tokenizer=self.tokenizer,
                batch=batch,
                policy_model_config=self.config,
                use_attention_sequence_id=self.model.transformer.
                attn_uses_sequence_id, # type: ignore
                return_last=self.return_last,
                return_lm_logits=self.return_lm_logits,
            )

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: Optional[SequenceClassifierOutput] = None,
    ) -> dict[str, torch.Tensor]:
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs: SequenceClassifierOutput,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return pairwise_loss(
            outputs,
            batch,
            self.loss_type,
        )
