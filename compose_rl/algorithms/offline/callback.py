# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""DPO Algorithm."""

import copy
from typing import Optional

import torch
from composer import Trainer
from composer.callbacks import LoadCheckpoint
from composer.core import State, get_precision_context
from composer.loggers import Logger
from llmfoundry.interfaces import CallbackWithConfig
from llmfoundry.utils import build_composer_model
# pyright does not recognize process_init_device though it is a declared export
from llmfoundry.utils.config_utils import process_init_device  # type: ignore


class ReferencePolicyCallback(CallbackWithConfig):
    """Callback to run reference policy in offline RL.

    Args:
        train_config (dict): Training config passed to callback via foundry train.py as
            callback is registered under callbacks_with_config registry.
    """

    def __init__(
        self,
        train_config: dict,
    ):
        self.train_config = copy.deepcopy(train_config)
        self.reference_model = None

    def after_load(self, state: State, logger: Logger) -> None:
        model_config = self.train_config['model']
        init_context = process_init_device(
            model_config,
            self.train_config.get('fsdp_config'),
        )
        name = model_config.pop('name')
        self.reference_model = build_composer_model(
            name=name,
            cfg=model_config,
            tokenizer=state.model.tokenizer, # type: ignore
            init_context=init_context,
            master_weights_dtype=model_config.get('master_weights_dtype', None),
        )

        original_load_path = self.train_config.get('load_path', None)

        # The base model checkpoint may have been supplied by a LoadCheckpoint callback,
        # so we need to check and apply that checkpoint to the reference model.
        load_checkpoint_callbacks = [
            LoadCheckpoint(
                load_path=callback.load_path,
                load_weights_only=callback.load_weights_only,
                strict_model_weights=callback.strict_model_weights,
                ignore_keys=callback.ignore_keys,
                event=callback.event,
            )
            for callback in state.callbacks
            if isinstance(callback, LoadCheckpoint)
        ]

        # For HF checkpoint, load_path is unset and should be handled in llmfoundry code.
        # Create a Trainer object to load model into FSDP
        _ = Trainer(
            model=self.reference_model,
            parallelism_config={'fsdp': state.fsdp_config},
            precision=state.precision,
            load_weights_only=True,
            load_strict_model_weights=False,
            load_path=original_load_path,
            callbacks=load_checkpoint_callbacks,
        )

    def before_forward(self, state: State, logger: Logger) -> Optional[int]:
        # Before every batch we need to do a forwards pass over the reference model
        with get_precision_context(state.precision):
            with torch.no_grad():
                assert self.reference_model is not None
                reference_outputs = self.reference_model(state.batch)
                state.batch.update({
                    'ref_logp': reference_outputs['policy_logp'],
                    'raw_ref_logits': reference_outputs['raw_logits'],
                })


class PairwiseReferencePolicyCallback(ReferencePolicyCallback):
    """Callback to run reference policy in pairwise offline RL.

    Args:
        train_config (dict): Training config passed to callback via foundry train.py as
            callback is registered under callbacks_with_config registry.
    """

    def before_forward(self, state: State, logger: Logger) -> Optional[int]:
        # Before every batch we need to do a forwards pass over the reference model
        with get_precision_context(state.precision):
            with torch.no_grad():
                assert self.reference_model is not None
                reference_outputs = self.reference_model(state.batch)
                state.batch.update({
                    'ref_chosen': reference_outputs['policy_chosen_logp'],
                    'ref_rejected': reference_outputs['policy_rejected_logp'],
                })
