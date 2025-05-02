# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ShortResponseReward class."""

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from compose_rl.reward_learning import IncreasingNumbersReward


@pytest.fixture
def reward() -> IncreasingNumbersReward:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return IncreasingNumbersReward(reward=1.0, tokenizer=tokenizer)


def test_is_number() -> None:
    assert IncreasingNumbersReward.is_number('123') == True
    assert IncreasingNumbersReward.is_number('-123.45') == True
    assert IncreasingNumbersReward.is_number('abc') == False
    assert IncreasingNumbersReward.is_number('12a') == False


@pytest.mark.parametrize(
    'batch, expected_rewards',
    [
        (
            {
                'zero_rewards': torch.zeros((3, 5)),
                'raw_untokenized_texts': [
                    ('', '1 2 3 4 5'),
                    ('', '5 4 3 2 1'),
                    ('', 'no numbers here'),
                ],
                'generated_lens': torch.tensor([5, 5, 3]),
            },
            [(0, 4, 1.0), (1, 4, 0.2), (2, 2, 0.0)],
        ),
        (
            {
                'zero_rewards': torch.zeros((2, 6)),
                'raw_untokenized_texts': [
                    ('', '1 word 2 another 3'),
                    ('', 'a 1 b 3 c 2 d'),
                ],
                'generated_lens': torch.tensor([6, 6]),
            },
            [(0, 5, 0.6), (1, 5, 2 / 7)],
        ),
    ],
)
def test_increasing_numbers(
    reward: IncreasingNumbersReward,
    batch: dict[str, Any],
    expected_rewards: list[tuple[int, int, float]],
) -> None:
    result = reward(batch)
    assert result.shape == batch['zero_rewards'].shape
    for idx, pos, expected in expected_rewards:
        assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected


def test_call_increasing_numbers_invalid_input(
    reward: IncreasingNumbersReward,
) -> None:
    invalid_batch: dict[str, torch.Tensor] = {
        'zero_rewards': torch.zeros((2, 6)),
        'generated_lens': torch.tensor([6, 6]),
    }
    with pytest.raises(AssertionError):
        reward(invalid_batch)
