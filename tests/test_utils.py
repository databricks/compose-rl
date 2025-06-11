# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import pytest

from compose_rl.utils import mask_eos
from compose_rl.utils.utils import (
    masked_mean,
    sample_wise_masked_mean,
    get_entropies,
    get_token_entropies,
    get_sequence_entropies,
    get_batched_generated_values
)


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


def test_get_token_entropies_uniform_distribution():
    """Test token entropies when logits represent a uniform distribution."""
    # Create uniform logits (all values are the same)
    batch_size, seq_len, vocab_size = 2, 3, 5
    uniform_logits = torch.ones((batch_size, seq_len, vocab_size))
    
    # For uniform distribution with vocab_size options, entropy should be log(vocab_size)
    expected_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
    
    # Calculate token entropies
    token_entropies = get_token_entropies(uniform_logits)
    
    # Check shape and values
    assert token_entropies.shape == (batch_size, seq_len)
    assert torch.allclose(token_entropies, expected_entropy * torch.ones((batch_size, seq_len)), atol=1e-5)


def test_get_token_entropies_deterministic_distribution():
    """Test token entropies when logits represent a deterministic distribution."""
    # Create logits with one very large value (simulating deterministic distribution)
    batch_size, seq_len, vocab_size = 2, 3, 5
    logits = torch.zeros((batch_size, seq_len, vocab_size))
    
    # Set one token to have a very high logit value
    high_value = 100.0
    for b in range(batch_size):
        for s in range(seq_len):
            logits[b, s, 0] = high_value
    
    # For deterministic distribution, entropy should be very close to 0
    token_entropies = get_token_entropies(logits)
    
    # Check shape and values (should be close to zero)
    assert token_entropies.shape == (batch_size, seq_len)
    assert torch.all(token_entropies < 1e-3)


def test_get_token_entropies_batch_variation():
    """Test token entropies with different distributions in the batch."""
    batch_size, seq_len, vocab_size = 3, 4, 5
    
    # Create different logit distributions for different items in the batch
    logits = torch.zeros((batch_size, seq_len, vocab_size))
    
    # Batch 0: Uniform distribution
    logits[0, :, :] = torch.ones((seq_len, vocab_size))
    
    # Batch 1: Deterministic distribution
    for s in range(seq_len):
        logits[1, s, s % vocab_size] = 100.0
    
    # Batch 2: Skewed distribution
    for s in range(seq_len):
        probs = torch.tensor([0.7, 0.1, 0.1, 0.05, 0.05])
        logits[2, s, :] = torch.log(probs)
    
    token_entropies = get_token_entropies(logits)
    
    # Check shape
    assert token_entropies.shape == (batch_size, seq_len)
    
    # Batch 0 should have high entropy (close to log(vocab_size))
    assert torch.allclose(token_entropies[0, :], torch.log(torch.tensor(vocab_size, dtype=torch.float)) * torch.ones(seq_len), atol=1e-5)
    
    # Batch 1 should have very low entropy (close to 0)
    assert torch.all(token_entropies[1, :] < 1e-3)
    
    # Batch 2 should have medium entropy
    # Calculate expected entropy for the skewed distribution
    probs = torch.tensor([0.7, 0.1, 0.1, 0.05, 0.05])
    expected_entropy = -torch.sum(probs * torch.log(probs))
    assert torch.allclose(token_entropies[2, :], expected_entropy * torch.ones(seq_len), atol=1e-5)


def test_get_sequence_entropies_full_mask():
    """Test sequence entropies when all tokens are masked in."""
    batch_size, seq_len, vocab_size = 2, 4, 5
    
    # Create different distributions for testing
    logits = torch.zeros((batch_size, seq_len, vocab_size))
    
    # Batch 0: Uniform distribution
    logits[0, :, :] = torch.ones((seq_len, vocab_size))
    
    # Batch 1: Varying entropy per position
    for s in range(seq_len):
        # Position-dependent concentration
        concentration = s + 1
        probs = torch.zeros(vocab_size)
        probs[0] = (concentration) / (concentration + vocab_size - 1)
        remainder = (1 - probs[0]) / (vocab_size - 1)
        probs[1:] = remainder
        logits[1, s, :] = torch.log(probs)
    
    # Full mask - all tokens included
    mask = torch.ones((batch_size, seq_len))
    
    sequence_entropies = get_sequence_entropies(logits, mask)
    
    # Check shape
    assert sequence_entropies.shape == (batch_size,)
    
    # Batch 0 should have entropy = log(vocab_size) since all positions have uniform distribution
    assert torch.allclose(sequence_entropies[0], torch.log(torch.tensor(vocab_size, dtype=torch.float)), atol=1e-5)
    
    # Batch 1 should have the average of position-dependent entropies
    token_entropies = get_token_entropies(logits[1:2])
    expected_avg_entropy = token_entropies.mean()
    assert torch.allclose(sequence_entropies[1], expected_avg_entropy, atol=1e-5)


def test_get_sequence_entropies_partial_mask():
    """Test sequence entropies with partial masking."""
    batch_size, seq_len, vocab_size = 2, 4, 5
    
    # Create logits with different entropy patterns
    logits = torch.zeros((batch_size, seq_len, vocab_size))
    
    # Fill with some arbitrary but different values
    for b in range(batch_size):
        for s in range(seq_len):
            concentration = b * seq_len + s + 1
            probs = torch.zeros(vocab_size)
            probs[0] = (concentration) / (concentration + vocab_size - 1)
            remainder = (1 - probs[0]) / (vocab_size - 1)
            probs[1:] = remainder
            logits[b, s, :] = torch.log(probs)
    
    # Partial masks - different for each batch
    mask = torch.zeros((batch_size, seq_len))
    mask[0, 0] = 1  # Only first position for batch 0
    mask[0, 2] = 1  # And third position for batch 0
    mask[1, :] = 1  # All positions for batch 1
    
    # Get token entropies for verification
    token_entropies = get_token_entropies(logits)
    
    # Calculate expected sequence entropies manually
    expected_entropies = torch.zeros(batch_size)
    # Batch 0: average of positions 0 and 2
    expected_entropies[0] = (token_entropies[0, 0] + token_entropies[0, 2]) / 2
    # Batch 1: average of all positions
    expected_entropies[1] = token_entropies[1].mean()
    
    # Get sequence entropies
    sequence_entropies = get_sequence_entropies(logits, mask)
    
    # Check shape and values
    assert sequence_entropies.shape == (batch_size,)
    assert torch.allclose(sequence_entropies, expected_entropies, atol=1e-5)


def test_get_sequence_entropies_empty_mask():
    """Test sequence entropies with a completely masked-out sequence."""
    batch_size, seq_len, vocab_size = 2, 3, 5
    
    # Create arbitrary logits
    logits = torch.randn((batch_size, seq_len, vocab_size))
    
    # Empty mask for first batch, full mask for second
    mask = torch.zeros((batch_size, seq_len))
    mask[1, :] = 1
    
    sequence_entropies = get_sequence_entropies(logits, mask)
    
    # Check that the first batch gives 0 (due to empty mask and epsilon in denominator)
    # The epsilon in the denominator (1e-10) makes this close to 0 but not exactly 0
    assert sequence_entropies[0] < 1e-9
    
    # Check second batch is properly calculated
    token_entropies = get_token_entropies(logits)
    expected_entropy = token_entropies[1].mean()
    assert torch.allclose(sequence_entropies[1], expected_entropy, atol=1e-5)


def test_get_sequence_entropies_single_item_batch():
    """Test sequence entropies with a single-item batch."""
    seq_len, vocab_size = 3, 5
    
    # Create logits for a single item
    logits = torch.zeros((1, seq_len, vocab_size))
    
    # Create a distribution with varying entropy per position
    for s in range(seq_len):
        probs = torch.zeros(vocab_size)
        probs[0] = 0.5
        probs[1:] = 0.5 / (vocab_size - 1)
        logits[0, s, :] = torch.log(probs)
    
    # Full mask
    mask = torch.ones((1, seq_len))
    
    # Calculate expected entropy
    expected_entropy = -(0.5 * torch.log(torch.tensor(0.5)) + 
                         (vocab_size - 1) * (0.5 / (vocab_size - 1)) * 
                         torch.log(torch.tensor(0.5 / (vocab_size - 1))))
    
    sequence_entropies = get_sequence_entropies(logits, mask)
    
    # Check shape and value
    assert sequence_entropies.shape == (1,)
    assert torch.allclose(sequence_entropies[0], expected_entropy, atol=1e-5)


@pytest.fixture
def mock_batched_generated_values(monkeypatch):
    """Mock for get_batched_generated_values to test get_entropies independently."""
    def mock_fn(batched_values, prompt_len, max_gen_len):
        # For test purposes, just return a tensor with the right shape
        batch_size = batched_values.size(0)
        gen_len = max_gen_len if isinstance(max_gen_len, int) else max_gen_len.item()
        vocab_size = batched_values.size(2)
        return torch.randn((batch_size, gen_len, vocab_size))
    
    monkeypatch.setattr(
        'compose_rl.utils.utils.get_batched_generated_values',
        mock_fn
    )


@pytest.fixture
def mock_sequence_entropies(monkeypatch):
    """Mock for get_sequence_entropies to test get_entropies independently."""
    def mock_fn(logits, action_mask):
        # For test purposes, return entropy values based on action_mask
        batch_size = logits.size(0)
        return torch.tensor([1.0] * batch_size, dtype=torch.float)
    
    monkeypatch.setattr(
        'compose_rl.utils.utils.get_sequence_entropies',
        mock_fn
    )


def test_get_entropies_basic(mock_batched_generated_values, mock_sequence_entropies):
    """Test basic functionality of get_entropies."""
    batch_size, seq_len, vocab_size = 2, 10, 5
    max_gen_len = 5
    
    # Create test inputs
    logits = torch.randn((batch_size, seq_len, vocab_size))
    action_mask = torch.ones((batch_size, max_gen_len))
    prompt_len = torch.tensor([3, 5])
    
    # Call the function
    entropies = get_entropies(logits, action_mask, prompt_len, max_gen_len)
    
    # With our mock functions, we expect tensor([1.0, 1.0])
    assert entropies.shape == (batch_size,)
    assert torch.allclose(entropies, torch.ones(batch_size))


def test_get_entropies_integration():
    """Integration test for get_entropies without mocks."""
    batch_size, prompt_seq_len, gen_len, vocab_size = 2, 5, 3, 10
    max_gen_len = gen_len
    
    # Create logits for the whole sequence (prompt + generation)
    logits = torch.zeros((batch_size, prompt_seq_len + gen_len, vocab_size))
    
    # Fill with different distributions
    # For prompt tokens (these shouldn't matter except for the last prompt token)
    logits[:, :prompt_seq_len, :] = torch.randn((batch_size, prompt_seq_len, vocab_size))

    # IMPORTANT: get_batched_generated_values will extract tokens from prompt_len-1 to prompt_len+max_gen_len-1
    # This includes the last token of the prompt and excludes the last token of the generation
    # So we need to be careful about which indices we fill with test values

    for b in range(batch_size):
        # Last token of prompt (will be included in entropy calculation)
        if b == 0:
            # First batch: uniform distribution (high entropy) for last prompt token
            logits[b, prompt_seq_len - 1, :] = torch.zeros(vocab_size)
        else:
            # Second batch: peaky distribution (low entropy) for last prompt token
            logits[b, prompt_seq_len - 1, 0] = 10.0
            logits[b, prompt_seq_len - 1, 1:] = -2.0

        # First two tokens of generation (the third/last one won't be used in entropy calculation)
        for s in range(gen_len - 1):
            # Create distributions with different entropy patterns
            if b == 0:
                # First batch: uniform distribution (high entropy)
                logits[b, prompt_seq_len + s, :] = torch.zeros(vocab_size)
            else:
                # Second batch: peaky distribution (low entropy)
                logits[b, prompt_seq_len + s, 0] = 10.0
                logits[b, prompt_seq_len + s, 1:] = -2.0

    prompt_len = torch.tensor([prompt_seq_len] * batch_size)
    action_mask = torch.ones((batch_size, gen_len))
    
    # Calculate entropies
    entropies = get_entropies(logits, action_mask, prompt_len, max_gen_len)
    
    # First batch should have high entropy (close to log(vocab_size))
    # Second batch should have calculated low entropy
    assert entropies.shape == (batch_size,)
    assert entropies[0] > entropies[1]
    assert torch.isclose(entropies[0], torch.log(torch.tensor(vocab_size, dtype=torch.float)), atol=0.1)
    
    # Calculate expected entropy for the peaky distribution in batch 1
    # Softmax of [10.0, -2.0, -2.0, ..., -2.0]
    peaky_logits = torch.zeros(vocab_size)
    peaky_logits[0] = 10.0
    peaky_logits[1:] = -2.0
    peaky_probs = F.softmax(peaky_logits, dim=0)
    expected_entropy = -torch.sum(peaky_probs * torch.log(peaky_probs))
    assert torch.isclose(entropies[1], expected_entropy, atol=1e-5)
