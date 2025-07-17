# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch

from compose_rl.data.dataloader import get_num_tokens_in_batch_online


def test_get_num_tokens_with_action_mask_and_prompt_len():
    """Test token counting with action_mask and prompt_len.

    Verifies correct counting of prompt + valid generated tokens.
    """
    batch = {
        'prompt_len':
            torch.tensor([4, 4]),  # Both prompts are 4 tokens
        'action_mask':
            torch.tensor([
                [1, 1, 1, 0, 0],  # 3 valid generated tokens
                [1, 1, 0, 0, 0],  # 2 valid generated tokens
            ]),
    }

    result = get_num_tokens_in_batch_online(batch, pad_token_id=0)
    expected = 4 + 4 + 3 + 2  # prompt_tokens + generated_tokens
    assert result == expected
