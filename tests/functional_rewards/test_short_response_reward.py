# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ShortResponseReward class."""

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from compose_rl.algorithms.reward_modeling import ShortResponseReward


@pytest.fixture
def reward() -> ShortResponseReward:
    config: dict[str, Any] = {
        'reward': 1.0,
        'len_threshold': 5,
    }
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return ShortResponseReward(tokenizer=tokenizer, **config)


@pytest.mark.parametrize(
    'batch, expected_rewards',
    [
        (
            {
                'zero_rewards': torch.zeros((3, 6)),
                'generated_lens': torch.tensor([3, 5, 6]),
            },
            [(0, 2, 1.0), (1, 4, 1.0), (2, 5, 0.0)],
        ),
        (
            {
                'zero_rewards': torch.zeros((2, 10)),
                'generated_lens': torch.tensor([10, 4]),
            },
            [(0, 9, 0.0), (1, 3, 1.0)],
        ),
    ],
)
def test_short_response(
    reward: ShortResponseReward,
    batch: dict[str, torch.Tensor],
    expected_rewards: list[tuple[int, int, float]],
) -> None:
    result = reward(batch)
    assert result.shape == batch['zero_rewards'].shape
    for idx, pos, expected in expected_rewards:
        assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected


def test_call_short_response_invalid_input(reward: ShortResponseReward) -> None:
    invalid_batch: dict[str, torch.Tensor] = {
        'zero_rewards': torch.zeros((2, 6)),
    }
    with pytest.raises(AssertionError):
        reward(invalid_batch)
