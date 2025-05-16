# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import warnings
from functools import partial
from typing import Any, Optional, Union, cast

import pytest
import torch
from composer import Trainer
from composer.core.precision import get_precision_context
from composer.optim import DecoupledAdamW
from composer.utils import dist
from llmfoundry.utils import build_tokenizer
from llmfoundry.utils.builders import build_composer_model
from llmfoundry.utils.config_utils import process_init_device  # type: ignore
from llmfoundry.utils.config_utils import (
    to_dict_container,
)
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention

from compose_rl.data import (
    finegrained_preference_dataset_collate_fn,
    pairwise_preference_dataset_collate_fn,
)
from compose_rl.reward_learning.hf_utils import AutoModelForCausalLMWithRM
from tests.common import FineGrainedPreference, PairwisePreference, world_size


def get_config(
    conf_path: str = './yamls/testing.yaml',
) -> DictConfig:
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return cast(DictConfig, test_cfg)


def _load_tokenizer_cfg(cfg: Union[dict[str, Any], DictConfig]) -> dict:
    if isinstance(cfg, DictConfig):
        config = to_dict_container(cfg)
    else:
        assert isinstance(cfg, dict)
        config = cfg
    return config


def _get_objs(
    request: pytest.FixtureRequest,
    conf_path: str = './yamls/testing.yaml',
    model_config_overrides: Optional[dict] = None,
    attn_impl: str = 'torch',
    is_hf: bool = False,
):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property',
    )
    test_cfg = get_config(conf_path=conf_path)
    if model_config_overrides is not None:
        for k, v in model_config_overrides.items():
            test_cfg.model[k] = v

    # Read FSDP Config as a dict
    fsdp_config = test_cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(
        fsdp_config,
        resolve=True,
    ) if fsdp_config else None

    # Check if we are running on GPU
    is_gpu = False
    for item in request.session.items:
        is_gpu |= item.get_closest_marker('gpu') is not None

    # Build Model
    # For fast initialization, use `meta` device
    if not is_hf:
        device = 'cuda' if is_gpu else 'cpu'
    else:
        device = 'cpu'
    test_cfg.precision = 'amp_bf16' if is_gpu else 'fp32'
    if not is_hf:
        test_cfg.model.attn_config = {
            'attn_impl': attn_impl,
        }
    test_cfg.model.init_device = device
    test_cfg.device = device

    test_cfg.global_train_batch_size = 2
    test_cfg.device_eval_batch_size = 2
    test_cfg.device_train_microbatch_size = 2

    tokenizer_cfg: dict[str, Any] = _load_tokenizer_cfg(test_cfg.tokenizer)
    tokenizer = build_tokenizer(
        test_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    name = test_cfg.model.pop('name')
    model = build_composer_model(
        name=name,
        cfg=to_dict_container(test_cfg.model),
        tokenizer=tokenizer,
    )
    model.model.lm_backbone = model.model.lm_backbone.to('cuda')  # type: ignore
    model.model.value_head = model.model.value_head.to('cuda')  # type: ignore

    # Optimizer
    assert test_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )
    return test_cfg, model, optimizer


def gen_random_batch(
    batch_size: int,
    test_cfg: Union[DictConfig, ListConfig],
    inputs: Optional[list[str]] = None,
):
    # inputs can be [], ['input_ids'], ['input_ids', 'inputs_embeds'], and ['inputs_embeds']
    # default to only input ids
    if inputs == None:
        inputs = ['input_ids']
    device = 'cuda'
    # generate input batch of random data, suitable for a Causal LM
    batch = {}
    batch['input_ids'] = torch.randint(
        low=0,
        #high=test_cfg.model.vocab_size,
        high=30000,
        size=(batch_size, test_cfg.max_seq_len * 2),
    ).to(device)
    batch['attention_mask'] = torch.ones(
        size=(batch_size, test_cfg.max_seq_len * 2),
        dtype=torch.int64,
    ).to(device)
    batch['chosen_len'] = (
        torch.ones(
            size=(batch_size,),
            dtype=torch.int64,
        ) * test_cfg.max_seq_len
    ).to(device)
    batch['rejected_len'] = (
        torch.ones(
            size=(batch_size,),
            dtype=torch.int64,
        ) * test_cfg.max_seq_len
    ).to(device)
    return batch


def test_forward_backward_hf_automodel():
    model_id = 'jdchang/llama3-small'
    sample_text = ['Welcome to compose RL!']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLMWithRM.from_pretrained(model_id)
    model.train()
    original_params = next(model.parameters()).clone().data
    optimizer = DecoupledAdamW(
        model.parameters(),
    )
    model_inputs = tokenizer(sample_text, return_tensors='pt')
    output = model(**model_inputs)
    loss = output.scores.mean()
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


@pytest.mark.gpu
@pytest.mark.world_size(2)
@pytest.mark.parametrize(
    'conf_path',
    [
        'tests/yamls/testing_hf.yaml',
        pytest.param(
            'tests/yamls/testing_hf_classifier.yaml',
            marks=pytest.mark.skip(
                reason=
                'TODO: reenable. temporarily skipping to turn GPU CI back on.'
            )
        ),
    ],
)
def test_forward_backward(
    request: pytest.FixtureRequest,
    conf_path: str,
    batch_size: int = 2,
):
    test_cfg, model, optimizer = _get_objs(
        request=request,
        conf_path=conf_path,
        is_hf=True,
    )
    batch = gen_random_batch(batch_size, test_cfg)

    assert batch['input_ids'].shape == torch.Size([
        batch_size,
        test_cfg.max_seq_len * 2,
    ])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)['total']  # type: ignore
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


@pytest.mark.skip(
    reason='TODO: reenable. temporarily skipping to turn GPU CI back on.'
)
@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_config', [None, {}])
@pytest.mark.parametrize(
    'model_params',
    [
        (
            'hf_pairwise_rm',
            PairwisePreference,
            pairwise_preference_dataset_collate_fn,
        ),
        (
            'hf_classifier_rm',
            FineGrainedPreference,
            finegrained_preference_dataset_collate_fn,
        ),
    ],
)
def test_hf_train(
    world_size: int,
    model_params: tuple[str, type[PairwisePreference], Any],
    fsdp_config: dict[str, Any],
):
    model_type, dataset_cls, collate_fn = model_params
    model_name = 'jdchang/llama3-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='[PAD]')
    max_seq_len = 10

    dataset = dataset_cls(size=32, max_seq_len=max_seq_len)

    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer,
            max_seq_len,
        ),
        sampler=dist.get_sampler(dataset),
        batch_size=2,
    )
    model_config = {
        'pretrained_model_name_or_path': model_name,
        'pretrained': True,
        'init_device': 'mixed',
        'use_flash_attention_2': True,
        'tokenizer': tokenizer,
        'return_last': True,
        'return_lm_logits': False,
    }

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'cpu_offload': False,
        'mixed_precision': 'PURE',
        'activation_checkpointing': True,
        'activation_cpu_offload': False,
        'verbose': True,
        'sync_module_states': True,
    }
    init_context = process_init_device(model_config, fsdp_config)
    model = build_composer_model(
        name=model_type,
        tokenizer=tokenizer,
        init_context=init_context,
        cfg=model_config,
    )
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=3e-4,
    )
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        parallelism_config={'fsdp': fsdp_config},
        max_duration='1ep',
    )
    print(trainer.state.model)
    with FSDP.summon_full_params(
        trainer.state.model,
        writeback=False,
        recurse=False,
    ):
        original_params = next(trainer.state.model.parameters()).clone().data

    trainer.fit()

    with FSDP.summon_full_params(
        trainer.state.model,
        writeback=False,
        recurse=False,
    ):
        updated_params = next(trainer.state.model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


@pytest.mark.gpu
@world_size(2)
def test_flashattention2(world_size: int):
    model_name = 'jdchang/llama3-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    max_seq_len = 5
    dataset = PairwisePreference(size=2, max_seq_len=max_seq_len)
    dataloader = DataLoader(
        dataset,
        collate_fn=partial(
            pairwise_preference_dataset_collate_fn,
            tokenizer,
            max_seq_len,
        ),
        sampler=dist.get_sampler(dataset),
        batch_size=2,
    )
    model_config_flash = {
        'pretrained_model_name_or_path': model_name,
        'pretrained': True,
        'init_device': 'cpu',
        'use_flash_attention_2': True,
        'tokenizer': tokenizer,
        'return_last': True,
        'return_lm_logits': False,
    }
    model_config = {
        'pretrained_model_name_or_path': model_name,
        'pretrained': True,
        'init_device': 'cpu',
        'use_flash_attention_2': False,
        'tokenizer': tokenizer,
        'return_last': True,
        'return_lm_logits': False,
    }

    init_context = process_init_device(model_config, {})
    model = build_composer_model(
        name='hf_pairwise_rm',
        tokenizer=tokenizer,
        init_context=init_context,
        cfg=model_config,
    )
    model_flash = build_composer_model(
        name='hf_pairwise_rm',
        tokenizer=tokenizer,
        init_context=init_context,
        cfg=model_config_flash,
    ).to('cuda')

    transformer_block = model_flash.model.lm_backbone.model.layers[  # type: ignore
        0]
    # Checks that Flash Attention has been properly initialized
    assert isinstance(
        transformer_block.self_attn,  # type: ignore
        LlamaAttention,
    )
    assert model.model.config._attn_implementation == (  # type: ignore
        'flash_attention_2'
    )

    with get_precision_context('amp_bf16'):
        for batch in dataloader:
            out = model.forward(batch)
            batch = {k: v.to('cuda') for k, v in batch.items()}
            out_flash = model_flash.forward(batch)
            out_flash = {k: v.to('cpu') for k, v in out_flash.items()}

            assert torch.all(
                out['chosen_scores'].bfloat16()
                != out['rejected_scores'].bfloat16(),
            )
            assert torch.all(
                out_flash['chosen_scores'].bfloat16()
                != out_flash['rejected_scores'].bfloat16(),
            )
