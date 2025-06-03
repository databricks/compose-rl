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
from composer.models.huggingface import HuggingFaceModel
from composer.utils.checkpoint import load_checkpoint
from llmfoundry.interfaces import CallbackWithConfig
from llmfoundry.utils import build_composer_model
# pyright does not recognize process_init_device though it is a declared export
from llmfoundry.utils.config_utils import process_init_device  # type: ignore


class DPOCallback(CallbackWithConfig):
    """Callback to run DPO in an offline RL setting.

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

        # For HF checkpoint, load_path is unset and should be handled in llmfoundry code.
        # Create a Trainer object to load model into FSDP
        fake_trainer = Trainer(
            model=self.reference_model,
            parallelism_config={'fsdp': state.fsdp_config},
            precision=state.precision,
            load_weights_only=True,
            load_strict_model_weights=False,
            load_path=original_load_path,
        )

        # The base model checkpoint may have been supplied by a LoadCheckpoint callback,
        # so we need to check and apply that checkpoint to the reference model.
        load_checkpoint_callbacks = [
            callback for callback in state.callbacks
            if isinstance(callback, LoadCheckpoint)
        ]

        if original_load_path is not None and len(
            load_checkpoint_callbacks,
        ) > 0:
            raise ValueError(
                'Cannot use `load_path` in the train config when using `LoadCheckpoint` callback. '
                + 'Please remove `load_path` from the train config.',
            )

        # For any LoadCheckpoint callbacks we found, we will load the checkpoint into the reference model.
        # If none are found, this for loop is a no-op.
        for load_checkpoint_callback in load_checkpoint_callbacks:
            assert isinstance(self.reference_model, HuggingFaceModel)

            # If using PEFT, we need to _not_ filter the state dict to only include the PEFT weights.
            # This is so the checkpoint can load the base model weights. Since the reference model is
            # not being update, we don't need to respect the `should_save_peft_only` flag from the original model
            # and can just hardcode it to False.
            self.reference_model.should_save_peft_only = False
            load_checkpoint(
                path=load_checkpoint_callback.parsed_path,
                state=fake_trainer.state,
                logger=logger,
                object_store=load_checkpoint_callback.load_object_store,
                strict_model_weights=load_checkpoint_callback.
                strict_model_weights,
                ignore_keys=load_checkpoint_callback.ignore_keys,
                load_weights_only=load_checkpoint_callback.load_weights_only,
            )

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
