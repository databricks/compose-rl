# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OutputLengthReward class."""

import pytest
import torch
from transformers import AutoTokenizer

from compose_rl.algorithms.reward_modeling import OutputLengthReward


@pytest.fixture
def reward() -> OutputLengthReward:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return OutputLengthReward(max_gen_len=10, tokenizer=tokenizer)


@pytest.mark.parametrize(
    'batch, expected_rewards',
    [
        (
            {
                'zero_rewards': torch.zeros((3, 10)),
                'generated_lens': torch.tensor([5, 8, 10]),
            },
            [(0, 4, 0.5), (1, 7, 0.8), (2, 9, 1.0)],
        ),
        (
            {
                'zero_rewards': torch.zeros((2, 15)),
                'generated_lens': torch.tensor([3, 15]),
            },
            [(0, 2, 0.3), (1, 14, 1.5)],
        ),
    ],
)
def test_call_output_length_reward(
    reward: OutputLengthReward,
    batch: dict[str, torch.Tensor],
    expected_rewards: list[tuple[int, int, float]],
) -> None:
    result = reward(batch)
    assert result.shape == batch['zero_rewards'].shape
    for idx, pos, expected in expected_rewards:
        assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected


def test_call_method_invalid_input(reward: OutputLengthReward) -> None:
    invalid_batch: dict[str, torch.Tensor] = {
        'zero_rewards': torch.zeros((2, 6)),
    }
    with pytest.raises(AssertionError):
        reward(invalid_batch)
