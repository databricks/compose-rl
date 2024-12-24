# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import pathlib
from functools import partial
from typing import Any

import pytest
import torch
from composer import Trainer
from composer.optim import DecoupledAdamW
from composer.utils import dist
from llmfoundry.models import ComposerHFCausalLM, ComposerMPTCausalLM
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from compose_rl.data import prompt_dataset_collate_fn
from compose_rl.ppo import (
    ComposerHFPolicyModel,
    ComposerMosaicPolicy,
    PPOCallback,
)
from tests.common import PromptDataset, world_size


@pytest.mark.parametrize('model_type', ['mpt', 'hf'])
def test_model_forward(
    tiny_gpt2_tokenizer: PreTrainedTokenizer,
    model_type: str,
):
    prompt_len = 10
    dataset = PromptDataset(prompt_len=prompt_len)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            prompt_dataset_collate_fn,
            tiny_gpt2_tokenizer,
            32,
        ),
        batch_size=2,
    )

    model = None
    if model_type == 'mpt':
        model_config = {
            'n_layers': 1,
            'attn_config': {
                'attn_impl': 'torch',
            },
            'tokenizer': tiny_gpt2_tokenizer,
        }
        model = ComposerMosaicPolicy(**model_config)
    elif model_type == 'hf':
        model_name = 'gpt2'
        model_config = {
            'tokenizer': tiny_gpt2_tokenizer,
            'pretrained_model_name_or_path': model_name,
            'pretrained': True,
        }
        model = ComposerHFPolicyModel(**model_config)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    for sample in dataloader:
        # Per sample we need to add in the arguments the callback would create
        sample['right_padded_attn_mask'] = torch.ones((2, 15))
        action_mask = torch.ones((2, 5))
        sample['action_mask'] = action_mask
        sample['max_gen_len'] = 5
        sample['actions'] = torch.ones((2, 5))
        sample['obs'] = torch.ones((2, 15)).to(dtype=torch.int64)

        model(sample)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_config', [{}])  # type: ignore
@pytest.mark.parametrize('model_type', ['mpt', 'hf'])
def test_ppo_train(
    tiny_gpt2_tokenizer: PreTrainedTokenizer,
    world_size: int,
    fsdp_config: dict[str, Any],
    model_type: str,
    tmp_path: pathlib.Path,
):

    max_seq_len = 32
    prompt_len = 10
    precision = 'amp_bf16'

    dataset = PromptDataset(prompt_len=prompt_len)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            prompt_dataset_collate_fn,
            tiny_gpt2_tokenizer,
            max_seq_len,
        ),
        sampler=dist.get_sampler(dataset),
        batch_size=4,
    )

    # We need to mock this method, since our dataset isn't a StreamingDataset
    dataloader.state_dict = lambda: {}
    dataloader.load_state_dict = lambda x: None

    tmp_model = None
    model_config = None
    # Making a reference model so we can make sure the KL is 0
    if model_type == 'mpt':
        model_config = {
            'name': 'mpt_causal_lm',
            'n_layers': 1,
            'attn_config': {
                'attn_impl': 'torch',
            },
            'tokenizer': tiny_gpt2_tokenizer,
        }

        tmp_model = ComposerMPTCausalLM(**model_config)
    elif model_type == 'hf':
        model_name = 'gpt2'
        model_config = {
            'tokenizer': tiny_gpt2_tokenizer,
            'pretrained_model_name_or_path': model_name,
            'pretrained': True,
            'use_flash_attention_2': True,
            'allow_embedding_resizing': True,
        }
        tmp_model = ComposerHFCausalLM(**model_config)
    else:
        raise ValueError(f'Unknown model type: {model_type}.')

    tmp_optimizer = DecoupledAdamW(tmp_model.parameters(), lr=1e-6)

    tmp_ref_path = str(tmp_path / 'ref_checkpoints')

    temp_dataloader = [{
        'input_ids': torch.ones((2, 15)).to(dtype=torch.int64),
        'attention_mask': torch.ones((2, 15)),
        'labels': torch.ones((2, 15)).to(dtype=torch.int64),
    }]

    temp_trainer = Trainer(
        model=tmp_model,
        train_dataloader=temp_dataloader,
        optimizers=tmp_optimizer,
        max_duration='1ba',
        parallelism_config={'fsdp': fsdp_config},
        save_folder=tmp_ref_path,
        save_weights_only=True,
        device_train_microbatch_size=2,
    )

    temp_trainer.fit()

    # After making the reference model, we can proceed with the PPO training
    tmp_ref_path = os.path.join(tmp_ref_path, 'latest-rank0.pt')

    if model_type == 'mpt':
        model = ComposerMosaicPolicy(**model_config)
    elif model_type == 'hf':
        model = ComposerHFPolicyModel(**model_config)

    optimizer = DecoupledAdamW(model.parameters(), lr=1e-8)

    num_batches_per_update = 2

    ref_model_config = copy.deepcopy(model_config)
    if model_type == 'hf':
        ref_model_config['name'] = 'hf_causal_lm'

    variables = {
        'buffer': {
            'name': 'MinibatchRolloutBuffer',
            'max_buffer_size': num_batches_per_update,
        },
        'max_gen_len': 8,
        'gamma': 0.99,
        'lambda_gae': 0.95,
        'generation_kwargs': {
            'use_cache': True,
            'do_sample': False,
        },
        'kl_controller': {
            'init_kl_coef': 0.2,
            'target': 0.01,
            'horizon': 12800,
            'kl_ctl_type': 'adaptive',
        },
        'reference_model': {
            'model_config': ref_model_config,
            'precision': precision,
            'load_path': tmp_ref_path,
            'non_train_fsdp_config': fsdp_config,
        },
        'device_generate_batch_size': 2,
        'epoch_per_iteration': 1,
        'num_batches_per_update': num_batches_per_update,
        'rewards': {
            'output_length': {
                'reward_type': 'output_length',
                'max_gen_len': 10,
            },
        },
    }
    train_config = {
        'model': model_config,
        'fsdp_config': fsdp_config,
        'seed': 17,
        'variables': variables,
        'max_seq_len': max_seq_len,
        'global_train_batch_size': 2,
        'device_train_batch_size': 2,
        'device_train_microbatch_size': 1,
    }

    tmp_save_path = str(tmp_path / 'checkpoints')
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        callbacks=PPOCallback(train_config=copy.deepcopy(train_config)),
        train_dataloader=dataloader,
        precision=precision,
        parallelism_config={'fsdp': fsdp_config},
        max_duration='3iter',
        device_train_microbatch_size=1,
        load_path=tmp_ref_path,
        save_folder=tmp_save_path,
        save_interval='1iter',
    )

    trainer.fit(duration='1iter')

    # This is the KL assert that must be true if we are truly loading from the same model.
    # This is only true on the first iteration
    assert torch.allclose(
        trainer.state.loss['kl/ift_kl'], # pyright: ignore
        torch.tensor(0.0),
        atol=5e-5,
    )

    # Continue training for the remaining iterations
    if model_type == 'mpt':
        model = ComposerMosaicPolicy(**model_config)
    elif model_type == 'hf':
        model = ComposerHFPolicyModel(**model_config)

    optimizer = DecoupledAdamW(model.parameters(), lr=1e-8)
    trainer_2 = Trainer(
        model=model,
        optimizers=optimizer,
        callbacks=PPOCallback(train_config=copy.deepcopy(train_config)),
        train_dataloader=dataloader,
        precision=precision,
        parallelism_config={'fsdp': fsdp_config},
        max_duration='3iter',
        device_train_microbatch_size=1,
        save_folder=tmp_save_path,
        autoresume=True,
        python_log_level='debug',
    )

    # Continue training for the remaining iterations
    trainer_2.fit()
