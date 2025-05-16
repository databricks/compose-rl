# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""PPO Composer Policy implementations."""

import collections
import logging
from typing import Any, MutableMapping, Optional, Union

import torch
from composer.models import HuggingFaceModel
from composer.utils import dist, is_model_fsdp
from llmfoundry.models import ComposerHFCausalLM
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from compose_rl.ppo.modeling_hf import ComposerHFPolicy
from compose_rl.ppo.modeling_mpt import MPTForPolicy
from compose_rl.ppo.modeling_utils import (
    composer_online_rl_forward,
    online_rl_loss,
)
from compose_rl.ppo.policy_configuration import MPTPolicyConfig
from compose_rl.utils import (
    clear_mb_load_balancing_loss,
    get_mb_load_balancing_loss,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

log = logging.getLogger(__name__)


class ComposerMosaicPolicy(HuggingFaceModel):

    def __init__(
        self,
        tokenizer: Tokenizer,
        **kwargs: dict[str, Any],
    ):

        model = self.model_class(self.config_class(**kwargs))

        train_metrics = []
        eval_metrics = []

        self.running_stats = collections.defaultdict(lambda: [])

        super().__init__(
            model=model,
            tokenizer=tokenizer, # pyright: ignore
            metrics=train_metrics,
            eval_metrics=eval_metrics,
        )

        self.tokenizer = tokenizer
        self.policy_kl = []

        self.compute_kl_loss = kwargs.get('compute_kl_loss', True)
        self.target_kl = kwargs.get('target_kl', 0.1)

    @property
    def model_class(self) -> type[MPTForPolicy]:
        return MPTForPolicy

    @property
    def config_class(self) -> type[MPTPolicyConfig]:
        return MPTPolicyConfig

    def forward(self, batch: MutableMapping):
        clear_mb_load_balancing_loss(
            self.config,
            self.model.transformer,  # type: ignore
        )

        ret_val = composer_online_rl_forward(batch, self.model)

        lbl = get_mb_load_balancing_loss(
            self.config,
            self.model.transformer,  # type: ignore
        )

        ret_val['lbl'] = lbl

        return ret_val

    def eval_forward(self, batch: MutableMapping, outputs: MutableMapping):
        raise ValueError(
            'Eval forward is not supported for ComposerMosaicPolicy.',
        )

    def loss(self, outputs: MutableMapping, batch: MutableMapping):
        return_dict, kl_loss = online_rl_loss(
            outputs=outputs,
            batch=batch,
            loss_type='ppo',
            value_clip_range=self.config.value_clip_range,
            value_loss_weight=self.config.value_loss_weight,
            policy_clip_ratio=self.config.policy_clip_ratio,
            add_direct_kl_loss=self.config.compute_kl_loss,
            kl_estimator=self.config.kl_estimator,
            kl_clip_range=self.config.kl_clip_range,
        )

        self.policy_kl.append(kl_loss)

        return return_dict

    def determine_early_stop(self):
        local_policy_kl = torch.stack(self.policy_kl)
        avg_policy_kl = torch.mean(
            torch.cat(dist.all_gather_object(local_policy_kl)),
        )
        early_stop = False
        log.info(f'average policy kl is: {avg_policy_kl}')
        if avg_policy_kl > self.target_kl * 1.5:  # pyright: ignore
            early_stop = True
            log.info(f'Early stopping actor critic with kl: {avg_policy_kl}')
        self.policy_kl = []
        return early_stop

    def set_batch_stats(self, batch_stats: dict[str, Any]):
        self.batch_stats = batch_stats  # pyright: ignore


class ComposerHFPolicyModel(ComposerHFPolicy):

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        config_overrides: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):

        self.running_stats = collections.defaultdict(lambda: [])

        super().__init__(
            tokenizer=tokenizer,
            config_overrides=config_overrides,
            **kwargs,
        )

        self.tokenizer = tokenizer
        self.policy_kl = []

        self.compute_kl_loss = False
        self.target_kl = 0.1

        # TODO: This needs to be removed once the config is fixed.
        if config_overrides is not None:
            self.compute_kl_loss = config_overrides.get('compute_kl_loss')
            self.target_kl = config_overrides.get('target_kl')

        self.loss_type = kwargs.get('loss_type', 'ppo')

        # Validating the input types
        assert isinstance(self.compute_kl_loss, bool)
        assert isinstance(self.target_kl, float)

    def forward(self, batch: MutableMapping):
        ret_val = composer_online_rl_forward(
            batch,
            self.model,
            loss_type=self.loss_type,  # pyright: ignore
        )
        return ret_val

    def generate(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any):
        pad_token_id = kwargs.pop('pad_token_id', self.tokenizer.pad_token_id)

        # Note: it seems as if we need to summon FSDP parameters here to ensure that we don't break
        # the standard actor critic forward pass.
        if is_model_fsdp(self.model):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # Note: We need to use the FSDP.summon_full_params context manager here because the generate function
            # does not seem to gather the weights for the LM head. This solution works because the tied weights of the LM head
            # are in the root FSDP module, and are summoned by the below context manager. See https://github.com/pytorch/pytorch/issues/100069
            # for more info.
            # Note: We use recurse=False here so that we only summon full params for the LM head, not the entire model.
            with FSDP.summon_full_params(
                self.model,
                writeback=False,
                recurse=False,
            ):
                return self.model.generate(
                    input_ids=input_ids,
                    pad_token_id=pad_token_id,
                    **kwargs,
                )

        else:
            return self.model.generate(
                input_ids=input_ids,
                pad_token_id=pad_token_id,
                **kwargs,
            )

    def eval_forward(self, batch: MutableMapping, outputs: MutableMapping):
        raise ValueError(
            'Eval forward is not supported for ComposerHFPolicy.',
        )

    def loss(self, outputs: MutableMapping, batch: MutableMapping):
        return_dict, kl_loss = online_rl_loss(
            outputs=outputs,
            batch=batch,
            loss_type=self.loss_type,  # pyright: ignore
            value_clip_range=self.config.value_clip_range,
            value_loss_weight=self.config.value_loss_weight,
            policy_clip_ratio=self.config.policy_clip_ratio,
            add_direct_kl_loss=self.config.compute_kl_loss,
            kl_estimator=self.config.kl_estimator,
            kl_clip_range=self.config.kl_clip_range,
        )

        self.policy_kl.append(kl_loss)

        return return_dict

    def determine_early_stop(self):
        local_policy_kl = torch.stack(self.policy_kl)
        avg_policy_kl = torch.mean(
            torch.cat(dist.all_gather_object(local_policy_kl)),
        )
        early_stop = False
        log.info(f'average policy kl is: {avg_policy_kl}')
        if avg_policy_kl > self.target_kl * 1.5:  # pyright: ignore
            early_stop = True
            log.info(f'Early stopping actor critic with kl: {avg_policy_kl}')
        self.policy_kl = []
        return early_stop

    def set_batch_stats(self, batch_stats: dict[str, Any]):
        self.batch_stats = batch_stats


class ComposerHFCriticFreePolicyModel(ComposerHFCausalLM):
    """HF class wrapper for Critic Free Policy model."""
    default_train_metrics: tuple = ()
    default_eval_metrics: tuple = ()

    def __init__(
        self,
        loss_type: str = 'grpo',
        normalize_advantage: bool = True,
        length_normalize_policy_loss: bool = True,
        policy_clip_ratio: float = 0.15,
        policy_clip_high_ratio: float | None = None,
        compute_kl_loss: bool = True,
        target_kl: float = 0.1,
        kl_estimator: str = 'k3',
        kl_clip_range: float = 40.0,
        **kwargs: Any,
    ):
        """Initialize the ComposerHFCriticFreePolicyModel.

        Args:
            loss_type (str): The type of loss to use. Default: ``'grpo'``.
            normalize_advantage (bool): Whether to normalize the advantage. Default: ``True``.
            length_normalize_policy_loss (bool): Whether to length normalize the policy loss and KL loss. Default: ``True``.
            policy_clip_ratio (float): The policy clip ratio. Default: ``0.15``.
            policy_clip_high_ratio (float | None): The high policy clip ratio. Default: ``None`` uses policy_clip_ratio.
            compute_kl_loss (bool): Whether to compute KL loss. Default: ``True``.
            target_kl (float): The target KL value. Default: ``0.1``.
            kl_estimator (str): The KL estimator to use. Default: ``'k3'``.
            kl_clip_range (float): The KL clip range. Default: ``40.0``.
        """
        super().__init__(**kwargs)
        self.policy_kl = []
        self.loss_type = loss_type
        self.normalize_advantage = normalize_advantage
        self.length_normalize_policy_loss = length_normalize_policy_loss
        self.policy_clip_ratio = policy_clip_ratio
        self.policy_clip_high_ratio = policy_clip_high_ratio
        self.compute_kl_loss = compute_kl_loss
        self.target_kl = target_kl
        self.kl_estimator = kl_estimator
        self.kl_clip_range = kl_clip_range

    def forward(self, batch: MutableMapping):
        ret_val = composer_online_rl_forward(
            batch,
            self.model,
            loss_type=self.loss_type,
        )
        return ret_val

    def eval_forward(self, batch: MutableMapping, outputs: MutableMapping):
        raise ValueError(
            'Eval forward is not supported for HF Critic Free Policy.',
        )

    def loss(self, outputs: MutableMapping, batch: MutableMapping):
        return_dict, kl_loss = online_rl_loss(
            outputs=outputs,
            batch=batch,
            loss_type=self.loss_type,
            policy_clip_ratio=self.policy_clip_ratio,
            policy_clip_high_ratio=self.policy_clip_high_ratio,
            length_normalize_policy_loss=self.length_normalize_policy_loss,
            add_direct_kl_loss=self.compute_kl_loss,
            kl_estimator=self.kl_estimator,
            kl_clip_range=self.kl_clip_range,
        )

        self.policy_kl.append(kl_loss)
        return return_dict

    def determine_early_stop(self):
        local_policy_kl = torch.stack(self.policy_kl)
        avg_policy_kl = torch.mean(
            torch.cat(dist.all_gather_object(local_policy_kl)),
        )
        early_stop = False
        log.info(f'average policy kl is: {avg_policy_kl}')
        if avg_policy_kl > self.target_kl * 1.5:  # pyright: ignore
            early_stop = True
            log.info(f'Early stopping actor critic with kl: {avg_policy_kl}')
        self.policy_kl = []
        return early_stop

    def set_batch_stats(self, batch_stats: dict[str, Any]):
        self.batch_stats = batch_stats
