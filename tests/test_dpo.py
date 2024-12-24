# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from functools import partial
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from composer import Trainer
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledAdamW
from composer.utils import dist
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from compose_rl.data import pairwise_preference_dataset_collate_fn
from compose_rl.dpo import ComposerDPOLM
from compose_rl.dpo.callback import DPOCallback
from tests.common import PairwisePreference, world_size


def test_dpo_callback_forward(tiny_gpt2_tokenizer: PreTrainedTokenizer):
    # Build DataLoader
    max_seq_len = 10
    dataset = PairwisePreference(max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            pairwise_preference_dataset_collate_fn,
            tiny_gpt2_tokenizer,
            max_seq_len,
        ),
        batch_size=2,
    )
    batch = next(iter(dataloader))

    # Build callback
    model_config = {
        'n_layers': 1,
        'attn_config': {
            'attn_impl': 'torch',
        },
        'loss_fn': 'torch_crossentropy',
        'tokenizer': tiny_gpt2_tokenizer,
    }
    model = ComposerDPOLM(**model_config)
    model_config['name'] = 'mpt_dpo_lm'
    train_config = {
        'model': model_config,
        'fsdp_config': {},
        'seed': 17,
    }
    callback = DPOCallback(train_config=train_config)
    Trainer(
        model=model,
        callbacks=callback,
    )

    assert 'ref_chosen' not in batch
    assert 'ref_rejected' not in batch

    state = MagicMock()
    logger = MagicMock()
    state.batch = batch
    state.precision = 'amp_fp16'

    callback.before_forward(state, logger)

    assert 'ref_chosen' in state.batch
    assert 'ref_rejected' in state.batch


def test_model_forward(tiny_gpt2_tokenizer: PreTrainedTokenizer):
    max_seq_len = 10
    dataset = PairwisePreference(max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            pairwise_preference_dataset_collate_fn,
            tiny_gpt2_tokenizer,
            max_seq_len,
        ),
        batch_size=2,
    )
    model_config = {
        'n_layers': 1,
        'attn_config': {
            'attn_impl': 'torch',
        },
        'loss_fn': 'torch_crossentropy',
        'tokenizer': tiny_gpt2_tokenizer,
    }
    model = ComposerDPOLM(**model_config)
    for sample in dataloader:
        output = model(sample)
        assert output is not None


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_config', [None, {}])  # type: ignore
def test_train(
    tiny_gpt2_tokenizer: PreTrainedTokenizer,
    world_size: int,
    fsdp_config: dict[str, Any],
):
    max_seq_len = 10
    dataset = PairwisePreference(max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            pairwise_preference_dataset_collate_fn,
            tiny_gpt2_tokenizer,
            max_seq_len,
        ),
        sampler=dist.get_sampler(dataset),
        batch_size=2,
    )
    model_config = {
        'n_layers': 1,
        'attn_config': {
            'attn_impl': 'torch',
        },
        'tokenizer': tiny_gpt2_tokenizer,
    }
    model = ComposerDPOLM(**model_config)
    model_config['name'] = 'mpt_dpo_lm'
    fsdp_config = {}
    train_config = {
        'model': model_config,
        'fsdp_config': fsdp_config,
        'seed': 17,
    }
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        callbacks=DPOCallback(train_config=train_config),
        parallelism_config={'fsdp': fsdp_config},
        max_duration='1ep',
    )
    trainer.fit()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_config', [None, {}])
def test_checkpoint_reloading(
    tiny_gpt2_tokenizer: PreTrainedTokenizer,
    world_size: int,
    fsdp_config: Optional[dict[str, Any]],
    tmp_path: pathlib.Path,
):
    max_seq_len = 10
    dataset = PairwisePreference(max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            pairwise_preference_dataset_collate_fn,
            tiny_gpt2_tokenizer,
            max_seq_len,
        ),
        sampler=dist.get_sampler(dataset),
        batch_size=2,
    )
    model_config = {
        'n_layers': 1,
        'attn_config': {
            'attn_impl': 'torch',
        },
        'tokenizer': tiny_gpt2_tokenizer,
    }

    # Making a dummy reference model so we can make sure the KL is 0
    tmp_model = ComposerDPOLM(**model_config)
    tmp_optimizer = DecoupledAdamW(tmp_model.parameters(), lr=1e-6)
    model_config['name'] = 'mpt_dpo_lm'
    fsdp_config = {}
    parallelism_config = {'fsdp': fsdp_config}

    init_checkpoint_dir = str(tmp_path / 'init_checkpoint')

    temp_trainer = Trainer(
        model=tmp_model,
        train_dataloader=dataloader,
        optimizers=tmp_optimizer,
        max_duration='1ba',
        parallelism_config=parallelism_config,
        save_folder=init_checkpoint_dir,
        save_weights_only=True,
        device_train_microbatch_size=2,
    )

    temp_trainer.fit()
    temp_trainer.close()

    # After making the reference model, we can proceed with the DPO training
    init_checkpoint_path = os.path.join(init_checkpoint_dir, 'latest-rank0.pt')

    model = ComposerDPOLM(**model_config)
    # Add more model_config specific to DPO
    model_config['name'] = 'mpt_dpo_lm'
    model_config['loss_type'] = 'dpo'
    model_config['beta'] = 0.1
    model_config['sft_alpha'] = 0.2

    train_config = {
        'model': model_config,
        'fsdp_config': fsdp_config,
        'seed': 17,
        # In a real run, this path would be set in the yaml config
        'load_path': init_checkpoint_path,
    }
    # Track the logged metrics from the trainer in a variable
    in_memory_logger = InMemoryLogger()
    new_save_folder = str(tmp_path / 'new_save_folder')
    trainer1 = Trainer(
        model=model,
        train_dataloader=dataloader,
        loggers=in_memory_logger,
        callbacks=DPOCallback(train_config=train_config),
        parallelism_config={'fsdp': fsdp_config},
        max_duration='8ba',
        autoresume=True,
        save_folder=new_save_folder,
        save_ignore_keys=['state/optimizers/*'],
        load_path=init_checkpoint_path,
    )
    # Run trainer 1 for partial duration for intermediate checkpoint
    trainer1.fit(duration='4ba')
    margins = in_memory_logger.data['loss/train/margin']
    # The first margin should be 0.0
    assert margins[0][
        1] == 0.0, 'The margin should be 0.0 in the first trainer fit'

    # Restart the training from the intermediate checkpoint
    in_memory_logger = InMemoryLogger()
    model = ComposerDPOLM(**model_config)
    trainer2 = Trainer(
        model=model,
        train_dataloader=dataloader,
        loggers=in_memory_logger,
        callbacks=DPOCallback(train_config=train_config),
        parallelism_config={'fsdp': fsdp_config},
        max_duration='8ba',
        save_overwrite=True,
        save_folder=new_save_folder,
        autoresume=True,
        load_path=init_checkpoint_path,
    )
    trainer2.fit()
    margins = in_memory_logger.data['loss/train/margin']
    # After resuming the training, the first margin should not be 0.0
    assert margins[0][
        1
    ] != 0.0, 'The margin should not be 0.0 in the second trainer fit after resuming'
