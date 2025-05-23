# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch

from compose_rl.utils import mask_eos
from compose_rl.utils.utils import masked_mean, sample_wise_masked_mean


def test_mask_eos_basic_functionality():
    # Create a simple test case with batch size 2, sequence length 10
    actions = torch.tensor([
        [1, 2, 3, 50, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 50, 17, 18, 19, 20],
    ])

    # right_padded_obs structure: [prompt tokens, action tokens, padding]
    right_padded_obs = torch.tensor([
        [101, 102, 103, 104, 105, 1, 2, 3, 50, 5, 6, 7, 8, 9,
         10],  # 5 prompt tokens + 10 action tokens
        [201, 202, 203, 204, 205, 11, 12, 13, 14, 15, 50, 17, 18, 19,
         20],  # 5 prompt tokens + 10 action tokens
    ])

    right_padded_attn_mask = torch.ones_like(right_padded_obs, dtype=torch.bool)

    prompt_len = torch.tensor([5, 5])  # Both prompts are 5 tokens long
    generated_len = torch.tensor([10, 10])  # Initial generated length is 10
    max_gen_len = 10
    eos_token_ids = [50]  # EOS token is 50
    pad_token = 999  # Pad token is 999

    # Call the function
    new_obs, new_attn_mask, new_gen_len, action_mask = mask_eos(
        actions,
        right_padded_obs,
        right_padded_attn_mask,
        prompt_len,
        generated_len,
        max_gen_len,
        eos_token_ids,
        pad_token,
    )

    # First sequence: EOS at index 3 -> length is 4
    assert new_gen_len[0].item() == 4
    # Second sequence: EOS at index 5 -> length is 6
    assert new_gen_len[1].item() == 6

    # 2. Action mask should have zeros after EOS
    expected_action_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ])
    assert torch.all(action_mask == expected_action_mask)

    # Right padded observation should have pad tokens after EOS
    # First sequence: 5 (prompt) + 4 (gen up to EOS) = index 9
    assert torch.all(new_obs[0, 9:] == pad_token)
    # Second sequence: 5 (prompt) + 6 (gen up to EOS) = index 11
    assert torch.all(new_obs[1, 11:] == pad_token)

    # Attention mask should be False after EOS
    assert torch.all(new_attn_mask[0, 9:] == False)
    assert torch.all(new_attn_mask[1, 11:] == False)


def test_mask_eos_no_eos():
    # Test case where no EOS tokens are found
    actions = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ])

    # right_padded_obs includes prompt + actions
    right_padded_obs = torch.tensor([
        [101, 102, 103, 104, 105, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [201, 202, 203, 204, 205, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ])

    right_padded_attn_mask = torch.ones_like(right_padded_obs, dtype=torch.bool)

    prompt_len = torch.tensor([5, 5])
    generated_len = torch.tensor([10, 10])
    max_gen_len = 10
    eos_token_ids = [50]
    pad_token = 999

    new_obs, new_attn_mask, new_gen_len, action_mask = mask_eos(
        actions,
        right_padded_obs,
        right_padded_attn_mask,
        prompt_len,
        generated_len,
        max_gen_len,
        eos_token_ids,
        pad_token,
    )

    # Check results - nothing should change
    assert torch.all(new_gen_len == generated_len)
    assert torch.all(action_mask == 1)
    assert torch.all(new_obs == right_padded_obs)
    assert torch.all(new_attn_mask == right_padded_attn_mask)


def test_mask_eos_multiple_eos_tokens():
    # Test with multiple possible EOS tokens
    actions = torch.tensor([
        [1, 2, 3, 50, 5, 6, 7, 8, 9,
         10],  # First sequence has EOS (50) at index 3
        [11, 12, 13, 14, 15, 51, 17, 18, 19,
         20],  # Second sequence has EOS (51) at index 5
    ])

    # right_padded_obs includes prompt + actions
    right_padded_obs = torch.tensor([
        [101, 102, 103, 104, 105, 1, 2, 3, 50, 5, 6, 7, 8, 9, 10],
        [201, 202, 203, 204, 205, 11, 12, 13, 14, 15, 51, 17, 18, 19, 20],
    ])

    right_padded_attn_mask = torch.ones_like(right_padded_obs, dtype=torch.bool)

    prompt_len = torch.tensor([5, 5])
    generated_len = torch.tensor([10, 10])
    max_gen_len = 10
    eos_token_ids = [50, 51]  # Multiple EOS tokens
    pad_token = 999

    new_obs, _, new_gen_len, action_mask = mask_eos(
        actions,
        right_padded_obs,
        right_padded_attn_mask,
        prompt_len,
        generated_len,
        max_gen_len,
        eos_token_ids,
        pad_token,
    )

    # Check results - both sequences should be masked after a certian point.
    assert new_gen_len[0].item() == 4
    assert new_gen_len[1].item() == 6

    expected_action_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ])
    assert torch.all(action_mask == expected_action_mask)

    # Check paddings in right_padded_obs
    assert torch.all(new_obs[0, 9:] == pad_token)  # 5 (prompt) + 4 (gen) = 9
    assert torch.all(new_obs[1, 11:] == pad_token)  # 5 (prompt) + 6 (gen) = 11


def test_mask_eos_eos_at_end():
    # Test with EOS at the end of the sequence
    actions = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 50],  # EOS at the very end
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # No EOS
    ])

    # right_padded_obs includes prompt + actions
    right_padded_obs = torch.tensor([
        [101, 102, 103, 104, 105, 1, 2, 3, 4, 5, 6, 7, 8, 9, 50],
        [201, 202, 203, 204, 205, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ])

    right_padded_attn_mask = torch.ones_like(right_padded_obs, dtype=torch.bool)

    prompt_len = torch.tensor([5, 5])
    generated_len = torch.tensor([10, 10])
    max_gen_len = 10
    eos_token_ids = [50]
    pad_token = 999

    new_obs, _, new_gen_len, action_mask = mask_eos(
        actions,
        right_padded_obs,
        right_padded_attn_mask,
        prompt_len,
        generated_len,
        max_gen_len,
        eos_token_ids,
        pad_token,
    )

    # Full length is maintained with EOS at the end
    assert new_gen_len[0].item() == 10
    assert new_gen_len[1].item() == 10  # No change

    # Since the EOS is at the very end, the action mask should not have any zeros
    # (assuming the function doesn't mask after EOS when EOS is the last token)
    expected_action_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    assert torch.all(action_mask == expected_action_mask)

    # No padding change since the entire sequence is used
    assert torch.all(new_obs == right_padded_obs)


def test_mask_eos_multiple_eos_same_sequence():
    # Test with multiple EOS tokens in the same sequence
    actions = torch.tensor([
        [1, 2, 50, 4, 50, 6, 7, 8, 9, 10],  # EOS at index 2 and 4
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # No EOS
    ])

    # right_padded_obs includes prompt + actions
    right_padded_obs = torch.tensor([
        [101, 102, 103, 104, 105, 1, 2, 50, 4, 50, 6, 7, 8, 9, 10],
        [201, 202, 203, 204, 205, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ])

    right_padded_attn_mask = torch.ones_like(right_padded_obs, dtype=torch.bool)

    prompt_len = torch.tensor([5, 5])
    generated_len = torch.tensor([10, 10])
    max_gen_len = 10
    eos_token_ids = [50]
    pad_token = 999

    new_obs, new_attn_mask, new_gen_len, action_mask = mask_eos(
        actions,
        right_padded_obs,
        right_padded_attn_mask,
        prompt_len,
        generated_len,
        max_gen_len,
        eos_token_ids,
        pad_token,
    )

    assert new_gen_len[0].item() == 3  # First EOS at index 2 -> length is 3
    assert new_gen_len[1].item() == 10  # No change

    expected_action_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    assert torch.all(action_mask == expected_action_mask)

    # Check padding
    assert torch.all(new_obs[0, 8:] == pad_token)  # 5 (prompt) + 3 (gen) = 8
    assert torch.all(new_attn_mask[0, 8:] == False)


def test_mask_eos_varying_prompt_lengths():
    # Test with different prompt lengths
    actions = torch.tensor([
        [1, 2, 3, 50, 5, 6, 7, 8, 9, 10],  # EOS at index 3
        [11, 12, 13, 14, 15, 50, 17, 18, 19, 20],  # EOS at index 5
    ])

    # right_padded_obs with different prompt lengths
    right_padded_obs = torch.tensor([
        [101, 102, 103, 1, 2, 3, 50, 5, 6, 7, 8, 9, 10, 999,
         999],  # 3 prompt tokens + 10 action tokens + padding
        [201, 202, 203, 204, 205, 206, 207, 11, 12, 13, 14, 15, 50, 17,
         18],  # 7 prompt tokens + 8 action tokens
    ])

    right_padded_attn_mask = torch.ones_like(right_padded_obs, dtype=torch.bool)
    # Set padding mask correctly
    right_padded_attn_mask[0, 13:] = False

    prompt_len = torch.tensor([3, 7])  # Different prompt lengths
    generated_len = torch.tensor([10, 8])  # Different generated lengths
    max_gen_len = 10
    eos_token_ids = [50]
    pad_token = 999

    new_obs, new_attn_mask, new_gen_len, action_mask = mask_eos(
        actions,
        right_padded_obs,
        right_padded_attn_mask,
        prompt_len,
        generated_len,
        max_gen_len,
        eos_token_ids,
        pad_token,
    )

    # First sequence: EOS at index 3 -> length is 4
    assert new_gen_len[0].item() == 4
    # Second sequence: EOS at index 5 -> length is 6
    assert new_gen_len[1].item() == 6

    # Check padding starts at the correct positions
    assert torch.all(new_obs[0, 7:] == pad_token)  # 3 (prompt) + 4 (gen) = 7
    assert torch.all(new_obs[1, 13:] == pad_token)  # 7 (prompt) + 6 (gen) = 13

    # Define expected attention mask pattern
    expected_attn_mask = torch.ones_like(right_padded_attn_mask)
    # First sequence: mask after prompt(3) + generated_len(4)
    expected_attn_mask[0, 7:] = False
    # Second sequence: mask after prompt(7) + generated_len(6)
    expected_attn_mask[1, 13:] = False

    # Check attention mask matches expected pattern
    assert torch.all(new_attn_mask == expected_attn_mask)

    # Check action mask as well
    expected_action_mask = torch.ones_like(actions)
    # First sequence: mask after EOS at index 3
    expected_action_mask[0, 4:] = 0
    # Second sequence: mask after EOS at index 5
    expected_action_mask[1, 6:] = 0

    assert torch.all(action_mask == expected_action_mask)


def test_sample_wise_masked_mean_basic():
    """Test basic functionality of sample_wise_masked_mean with simple cases."""
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    result = sample_wise_masked_mean(values, mask)

    # First sample mean: (1*1 + 2*1 + 3*0) / (1+1+0) = 3/2 = 1.5
    # Second sample mean: (4*1 + 5*0 + 6*1) / (1+0+1) = 10/2 = 5.0
    # Final result: (1.5 + 5.0) / 2 = 3.25
    expected = torch.tensor(3.25)

    assert torch.allclose(result, expected)


def test_sample_wise_masked_mean_all_valid():
    """Test when all values are valid (mask is all ones)."""
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    result = sample_wise_masked_mean(values, mask)
    global_mean_result = masked_mean(values, mask)

    # First sample mean: (1+2+3)/3 = 2.0
    # Second sample mean: (4+5+6)/3 = 5.0
    # Final result: (2.0 + 5.0) / 2 = 3.5
    expected = torch.tensor(3.5)

    assert torch.allclose(result, expected)
    assert torch.allclose(global_mean_result, expected)


def test_sample_wise_masked_mean_single_valid_per_sample():
    """Test when each sample has only one valid value."""
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result = sample_wise_masked_mean(values, mask)

    # First sample mean: 1.0/1 = 1.0
    # Second sample mean: 6.0/1 = 6.0
    # Final result: (1.0 + 6.0) / 2 = 3.5
    expected = torch.tensor(3.5)

    assert torch.allclose(result, expected)


def test_sample_wise_masked_mean_single_sample():
    """Test with a single sample."""
    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    result = sample_wise_masked_mean(values, mask)

    # Only one sample mean: (1+3)/2 = 2.0
    # Final result: 2.0
    expected = torch.tensor(2.0)

    assert torch.allclose(result, expected)
