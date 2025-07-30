# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Reward Model Composer Implementation."""

import logging
from contextlib import nullcontext
from functools import partial
from typing import Any, Mapping, MutableMapping, Optional, Union

import torch
from composer.distributed.shared_utils import get_summon_params_fn
from composer.utils import is_model_fsdp
from llmfoundry.models import ComposerHFCausalLM, ComposerMPTCausalLM

from compose_rl.algorithms.reward_modeling.base_reward import (
    RewardModel,
    Tokenizer,
)
from compose_rl.algorithms.reward_modeling.hf_utils import \
    SequenceClassifierOutput
from compose_rl.algorithms.reward_modeling.model_methods import (
    ClassifierRewardEnum,
    PairwiseRewardEnum,
    causal_classifier_forward,
    classifier_forward,
    classifier_loss,
    pairwise_forward,
    pairwise_loss,
)
from compose_rl.algorithms.reward_modeling.modeling_hf import \
    ComposerHFSequenceClassification
from compose_rl.algorithms.reward_modeling.modeling_mpt import \
    MPTForSequenceClassification

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


class ComposerHFCausalClassifierRewardModel(ComposerHFCausalLM, RewardModel):
    default_train_metrics: tuple = ()
    default_eval_metrics: tuple = ()

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        loss_type: str = 'bce',
        return_last: bool = True,
        should_reset_output_embed: bool = True,
        **kwargs: Any,
    ):
        self.loss_type = ClassifierRewardEnum(loss_type)
        self.return_last = return_last
        self.eos_token_id = tokenizer.eos_token_id  # type: ignore

        config_overrides = {}

        if 'config_overrides' in kwargs:
            config_overrides.update(kwargs.pop('config_overrides'))

        self.min_threshold = kwargs.pop('min_threshold', None)
        self.max_threshold = kwargs.pop('max_threshold', None)

        if tokenizer is None:
            raise ValueError('Tokenizer must be provided.')

        super().__init__(
            tokenizer=tokenizer,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            config_overrides=config_overrides,
            **kwargs,
        )

        self.reset_output_embed = False
        self.should_reset_output_embed = should_reset_output_embed

    def mask_last_embed_except_eos(
        self,
        fill_value: float = -100,
    ) -> None:
        """Mask out all but the last embedding for the EOS token."""
        logging.info('Resetting output embedding layer.')

        context_manager = nullcontext
        if is_model_fsdp(self.model):
            summon_full_params = get_summon_params_fn(self.model)
            context_manager = partial(
                summon_full_params,
                self.model,
                writeback=True,
                recurse=False,
            )

        with context_manager():
            mask = torch.full_like(
                self.model.lm_head.weight.data,  # type: ignore
                fill_value,  # type: ignore
            )  # type: ignore

            # Reset the values here? It might help
            mask[self.eos_token_id, :] = 0.0  # type: ignore

            with torch.no_grad():
                self.model.lm_head.weight.copy_(mask)  # type: ignore

        self.reset_output_embed = True
        logging.info('Finished resetting output embedding layer.')

    def forward(
        self,
        batch: MutableMapping,
    ) -> Union[dict[str, torch.Tensor], torch.Tensor]:
        if not self.reset_output_embed and self.should_reset_output_embed:
            self.mask_last_embed_except_eos()

        is_inference = batch.get('is_inference', False)
        if is_inference:
            # Inference code should be able to handle this arbitrary result
            logits = self.model(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
            ).logits
            logits = logits[:, :, self.eos_token_id]  # type: ignore
            if self.min_threshold is not None and self.max_threshold is not None:
                logits: torch.Tensor = torch.clamp(
                    logits,
                    min=self.min_threshold,
                    max=self.max_threshold,
                )
            return logits
        else:
            outputs = causal_classifier_forward(
                model=self.model,
                tokenizer=self.tokenizer,
                batch=batch,
            )
        return outputs

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: Optional[SequenceClassifierOutput] = None,
    ) -> dict[str, torch.Tensor]:
        return outputs if outputs is not None else self.forward(
            batch,
        )  # type: ignore

    def loss(self, outputs: SequenceClassifierOutput,
             batch: Mapping) -> dict[str, torch.Tensor]:
        return classifier_loss(
            outputs,
            batch,
            self.loss_type,
        )
