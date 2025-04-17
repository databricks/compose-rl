# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BadGenerationEndReward class."""

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from compose_rl.reward_learning import GSM8KAnswerVeriferReward


@pytest.fixture
def reward() -> GSM8KAnswerVeriferReward:
    config: dict[str, Any] = {}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return GSM8KAnswerVeriferReward(config, tokenizer)


def test_validate_config(reward: GSM8KAnswerVeriferReward) -> None:
    reward.validate_config()


def test_call_base_verifer_invalid_input(reward: GSM8KAnswerVeriferReward) -> None:
    invalid_batch: dict[str, torch.Tensor] = {
        'zero_rewards': torch.zeros((2, 6)),
        'generated_lens': torch.tensor([6, 6]),
    }
    with pytest.raises(AssertionError):
        reward(invalid_batch)


@pytest.mark.parametrize(
    'batch, expected_rewards',
    [
        (
            {
                'zero_rewards': torch.zeros((3, 5)),
                'raw_untokenized_texts': [
                    ('', 'Answer is ####24'),
                    ('', 'This is a very long string answer'),
                    ('', 'There are 32 initial numbers, adding 64 new elements gives 32+64 = 96. Answer is \boxed{96}'),
                ],
                'verified_answers': ['24', '89', '96'],
                'generated_lens': torch.tensor([5, 5, 3]),
            },
            [(0, 4, 1.0), (1, 4, 0.0), (2, 2, 1.0)],
        ),
        (
            {
                'zero_rewards': torch.zeros((2, 6)),
                'raw_untokenized_texts': [
                    ('', '####1'),
                    ('', '##2'),
                ],
                'verified_answers': ['1', '4'],
                'generated_lens': torch.tensor([6, 6]),
            },
            [(0, 5, 1.0), (1, 5, 0.0)],
        ),
    ],
)
def test_gms8k_answer_verifier(
    reward: GSM8KAnswerVeriferReward,
    batch: dict[str, Any],
    expected_rewards: list[tuple[int, int, float]],
) -> None:
    result = reward(batch)
    assert result.shape == batch['zero_rewards'].shape
    for idx, pos, expected in expected_rewards:
        assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected
