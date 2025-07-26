# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import os
import zipfile
from typing import Any

import pytest
import requests


## MODEL HELPERS ##
def causal_lm_model_helper(config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForCausalLM.from_config(config)


## CONFIG HELPERS ##
def tiny_gpt2_config_helper():
    pytest.importorskip('transformers')
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    config_dict = {
        'activation_function': 'gelu_new',
        'architectures': ['GPT2LMHeadModel',],
        'attn_pdrop': 0.1,
        'bos_token_id': 50256,
        'embd_pdrop': 0.1,
        'eos_token_id': 50256,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-05,
        'model_type': 'gpt2',
        'n_ctx': 1024,
        'n_embd': 64,
        'n_head': 2,
        'n_layer': 2,
        'n_positions': 1024,
        'resid_pdrop': 0.1,
        'summary_activation': None,
        'summary_first_dropout': 0.1,
        'summary_proj_to_labels': True,
        'summary_type': 'cls_index',
        'summary_use_proj': True,
        'task_specific_params': {
            'text-generation': {
                'do_sample': True,
                'max_length': 50,
            },
        },
        'vocab_size': 50258,
    }

    config_object = GPT2Config(
        **config_dict,
    )
    return config_object


def tiny_llama_config_helper():
    pytest.importorskip('transformers')
    from transformers.models.llama.configuration_llama import LlamaConfig
    config_dict = {
        'architectures': ['LlamaForCausalLM'],
        'bos_token_id': 1,
        'eos_token_id': 2,
        'hidden_act': 'silu',
        'hidden_size': 128,
        'intermediate_size': 256,
        'max_position_embeddings': 2048,
        'model_type': 'llama',
        'num_attention_heads': 4,
        'num_hidden_layers': 2,
        'num_key_value_heads': 4,
        'rms_norm_eps': 1e-06,
        'rope_theta': 10000.0,
        'use_cache': True,
        'vocab_size': 50258,  # Match GPT-2 tokenizer vocabulary size
    }

    config_object = LlamaConfig(
        **config_dict,
    )
    return config_object


def assets_path():
    rank = os.environ.get('RANK', '0')
    folder_name = 'tokenizers' + (f'_{rank}' if rank != '0' else '')
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'assets',
        folder_name,
    )


@pytest.fixture(scope='session')
def tokenizers_assets():
    download_tokenizers_files()


def download_tokenizers_files():
    """Download the tokenizers assets.

    We download from github, because downloading from HF directly is flaky and gets rate limited easily.

    Raises:
        ValueError: If the checksum of the downloaded file does not match the expected checksum.
    """
    # Define paths
    tokenizers_dir = assets_path()

    if os.path.exists(tokenizers_dir):
        return

    # Create assets directory if it doesn't exist
    os.makedirs(tokenizers_dir, exist_ok=True)

    # URL for the tokenizers.zip file
    url = 'https://github.com/mosaicml/ci-testing/releases/download/tokenizers/tokenizers.zip'
    expected_checksum = '12dc1f254270582f7806588f1f1d47945590c5b42dee28925e5dab95f2d08075'

    # Download the zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    zip_path = os.path.join(tokenizers_dir, 'tokenizers.zip')

    # Check the checksum
    checksum = hashlib.sha256(response.content).hexdigest()
    if checksum != expected_checksum:
        raise ValueError(
            f'Checksum mismatch: expected {expected_checksum}, got {checksum}',
        )

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file
    print(f'Extracting tokenizers.zip to {tokenizers_dir}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tokenizers_dir)

    # Optionally remove the zip file after extraction
    os.remove(zip_path)


## TOKENIZER HELPERS ##
def assets_tokenizer_helper(name: str, **kwargs: Any):
    """Load a tokenizer from the assets directory."""
    transformers = pytest.importorskip('transformers')

    download_tokenizers_files()

    assets_dir = assets_path()
    tokenizer_path = os.path.join(assets_dir, name)

    # Load the tokenizer
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        **kwargs,
    )
    return hf_tokenizer


## SESSION MODELS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_model(_session_tiny_gpt2_config):  # type: ignore
    return causal_lm_model_helper(_session_tiny_gpt2_config)


@pytest.fixture(scope='session')
def _session_tiny_llama_model(_session_tiny_llama_config):  # type: ignore
    return causal_lm_model_helper(_session_tiny_llama_config)


## SESSION CONFIGS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_config():  # type: ignore
    return tiny_gpt2_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_llama_config():  # type: ignore
    return tiny_llama_config_helper()


## SESSION TOKENIZERS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_tokenizer(tokenizers_assets):  # type: ignore
    tokenizer = assets_tokenizer_helper('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


## MODEL FIXTURES ##
@pytest.fixture
def tiny_gpt2_model(_session_tiny_gpt2_model):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_model)


@pytest.fixture
def tiny_llama_model(_session_tiny_llama_model):  # type: ignore
    return copy.deepcopy(_session_tiny_llama_model)


## TOKENIZER FIXTURES ##
@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)
