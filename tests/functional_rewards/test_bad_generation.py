# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BadGenerationEndReward class."""

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from compose_rl.reward_learning import BadGenerationEndReward


@pytest.fixture
def reward() -> BadGenerationEndReward:
    config: dict[str, Any] = {
        'reward': -1.0,
        'eos_penalty': True,
        'extra_special_tokens': ['<|im_end|>'],
    }
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b',)
    special_tokens_dict: dict[str, list[str]] = {
        'additional_special_tokens': ['<|im_end|>'],
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    return BadGenerationEndReward(config, tokenizer)


def test_validate_config(reward: BadGenerationEndReward) -> None:
    reward.validate_config()


def test_validate_config_missing_fields() -> None:
    with pytest.raises(AssertionError):
        BadGenerationEndReward(
            {},
            AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b'),
        )


@pytest.mark.parametrize(
    'batch, expected_rewards',
    [
        (
            {
                'zero_rewards':
                    torch.zeros((3, 5)),
                'seq_lens':
                    torch.tensor([5, 5, 5]),
                'input_ids':
                    torch.tensor([
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 0],
                        [1, 2, 3, 4, 50277],
                    ]),
                'generated_lens':
                    torch.tensor([5, 5, 5]),
            },
            [(0, 4, -1.0), (1, 4, 0.0), (2, 4, 0.0)],
        ),
    ],
)
def test_call_bad_generation(
    reward: BadGenerationEndReward,
    batch: dict[str, torch.Tensor],
    expected_rewards: list[tuple[int, int, float]],
) -> None:
    result = reward(batch)
    assert result.shape == batch['zero_rewards'].shape
    for idx, pos, expected in expected_rewards:
        assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected


def test_call_bad_generation_invalid_input(
    reward: BadGenerationEndReward,
) -> None:
    invalid_batch: dict[str, torch.Tensor] = {
        'zero_rewards': torch.zeros((2, 6)),
        'seq_lens': torch.tensor([6, 6]),
        'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]),
    }
    with pytest.raises(AssertionError):
        reward(invalid_batch)


def test_special_tokens_added(reward: BadGenerationEndReward) -> None:
    assert reward.tokenizer is not None
    assert '<|im_end|>' in reward.tokenizer.additional_special_tokens
    assert reward.tokenizer.additional_special_tokens_ids != []
