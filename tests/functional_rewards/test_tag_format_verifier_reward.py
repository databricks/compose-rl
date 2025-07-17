# Copyright 2025 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TagFormatVerifierReward class."""

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer


from compose_rl.algorithms.reward_modeling import TagFormatVerifierReward, Tokenizer


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(params=["think", "answer", "judge"])
def reward(request: pytest.FixtureRequest, tokenizer: Tokenizer) -> TagFormatVerifierReward:
    """Instantiate the reward once for each tag keyword."""
    return TagFormatVerifierReward(
        tokenizer=tokenizer,
        reward=1.0,
        tag_keyword=request.param,
    )



def test_call_base_verifier_invalid_input(reward: TagFormatVerifierReward) -> None:
    """Forward pass should raise if mandatory keys are missing / malformed."""
    invalid_batch = {
        "zero_rewards": torch.zeros((1, 3)),
        "generated_lens": torch.tensor([3]),
        # 'raw_untokenized_texts' key is intentionally absent
    }
    with pytest.raises(AssertionError):
        reward(invalid_batch)


@pytest.mark.parametrize(
    "tag_keyword,batch,expected_rewards",
    [
        (
            "think",
            {
                "zero_rewards": torch.zeros((4, 6)),
                "raw_untokenized_texts": [
                    ("", "Here is my <think>some reasoning</think> done."),          # valid
                    ("", "Here is my <think>some reasoning..."),                     # missing close tag
                    ("", "Here is my <think>   </think> just whitespace."),          # empty body
                    ("", "</think>oops<think>bad order</think>"),                    # end before start
                ],
                "verified_answers": ["", "", "", ""],  # not used by verifier
                "generated_lens": torch.tensor([6, 6, 6, 6]),
            },
            [(0, 5, 1.0), (1, 5, 0.0), (2, 5, 0.0), (3, 5, 0.0)],
        ),
        (
            "answer",
            {
                "zero_rewards": torch.zeros((3, 5)),
                "raw_untokenized_texts": [
                    ("", "Result: <answer>42</answer> ✅"),                           # valid
                    ("", "Result: </answer>"),                                        # closing only
                    ("", "<answer>   </answer>"),                                     # empty body
                ],
                "verified_answers": ["", "", ""],
                "generated_lens": torch.tensor([5, 5, 5]),
            },
            [(0, 4, 1.0), (1, 4, 0.0), (2, 4, 0.0)],
        ),
        (
            "judge",
            {
                "zero_rewards": torch.zeros((2, 4)),
                "raw_untokenized_texts": [
                    ("", "I am <judge>true</judge>."),                                # valid
                    ("", "I am <judge></judge> unsure."),                             # empty body
                ],
                "verified_answers": ["", ""],
                "generated_lens": torch.tensor([4, 4]),
            },
            [(0, 3, 1.0), (1, 3, 0.0)],
        ),
    ],
)
def test_tag_format_verifier(
    tag_keyword: str,
    batch: dict[str, Any],
    expected_rewards: list[tuple[int, int, float]],
    tokenizer: Tokenizer,
) -> None:
    """End‑to‑end check that rewards are placed correctly at the final token."""
    reward_fn = TagFormatVerifierReward(
        tokenizer=tokenizer,
        reward=1.0,
        tag_keyword=tag_keyword,
    )

    result = reward_fn(batch)

    # Shape must match the zero_rewards tensor
    assert result.shape == batch["zero_rewards"].shape

    # Verify each expected (row, col, value)
    for idx, pos, expected in expected_rewards:
        assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected
