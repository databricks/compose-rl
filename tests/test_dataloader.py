# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch

from compose_rl.data.dataloader import get_num_tokens_in_batch


def test_get_num_tokens_with_action_mask_and_prompt_len():
    """Test token counting with action_mask and prompt_len.

    Verifies correct counting of prompt + valid generated tokens.
    """
    batch = {
        'sequences':
            torch.tensor([
                [1, 2, 3, 4, 10, 11, 12, 0,
                 0],  # prompt: [1,2,3,4], generated: [10,11,12], padding: [0,0]
                [5, 6, 7, 8, 20, 21, 0, 0,
                 0],  # prompt: [5,6,7,8], generated: [20,21], padding: [0,0,0]
            ]),
        'prompt_len':
            torch.tensor([4, 4]),  # Both prompts are 4 tokens
        'action_mask':
            torch.tensor([
                [1, 1, 1, 0, 0],  # 3 valid generated tokens
                [1, 1, 0, 0, 0],  # 2 valid generated tokens
            ]),
    }

    result = get_num_tokens_in_batch(batch, pad_token_id=0)
    expected = 4 + 4 + 3 + 2  # prompt_tokens + generated_tokens
    assert result == expected


def test_get_num_tokens_fallback_without_action_mask():
    """Test fallback counting when action_mask is not available.

    Verifies fallback to counting all non-padding tokens.
    """
    batch = {
        'sequences':
            torch.tensor([
                [1, 2, 3, 4, 10, 11, 12, 0, 0],  # 7 non-padding tokens
                [5, 6, 7, 8, 20, 21, 0, 0, 0],  # 6 non-padding tokens
            ]),
    }

    result = get_num_tokens_in_batch(batch, pad_token_id=0)
    expected = 7 + 6
    assert result == expected
