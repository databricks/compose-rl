# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MATHVerifierReward class."""

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from compose_rl.algorithms.reward_modeling import MCQAVerifierReward


@pytest.fixture
def reward() -> MCQAVerifierReward:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return MCQAVerifierReward(reward=2.0, tokenizer=tokenizer)


def test_call_base_verifer_invalid_input(reward: MCQAVerifierReward) -> None:
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
                    ('', 'Final answer - \\boxed{C}'),
                    (
                        '',
                        '\nExact Answer :   E',
                    ),
                    (
                        '',
                        'Thus, final answer: **a**',
                    ),
                ],
                'verified_answers': ['C', 'D', 'A'],
                'generated_lens': torch.tensor([5, 5, 3]),
            },
            [(0, 4, 2.0), (1, 4, 0.0), (2, 2, 2.0)],
        ),
        (
            {
                'zero_rewards': torch.zeros((2, 6)),
                'raw_untokenized_texts': [
                    ('', 'Possible answers are A, B or C.'),
                    ('', 'The answer is (d).'),
                ],
                'verified_answers': ['B', 'D'],
                'generated_lens': torch.tensor([6, 6]),
            },
            [(0, 5, 2.0), (1, 5, 0.0)],
        ),
    ],
)
def test_mcqa_verifier(
    reward: MCQAVerifierReward,
    batch: dict[str, Any],
    expected_rewards: list[tuple[int, int, float]],
) -> None:
    result = reward(batch)
    assert result.shape == batch['zero_rewards'].shape
    for idx, pos, expected in expected_rewards:
        assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected
