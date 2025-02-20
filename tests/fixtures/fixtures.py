# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer


def tiny_gpt2_tokenizer_helper():
    hf_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    hf_tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    return hf_tokenizer


@pytest.fixture(scope='session')
def _session_tiny_gpt2_tokenizer():  # type: ignore
    return tiny_gpt2_tokenizer_helper()


@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer: PreTrainedTokenizer):
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)
