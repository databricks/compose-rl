# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import pytest

from compose_rl.utils.utils import (
    get_entropies,
    get_token_entropies,
    get_sequence_entropies,
    get_batched_generated_values
)


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
    # For prompt tokens (these shouldn't matter as they'll be filtered out)
    logits[:, :prompt_seq_len, :] = torch.randn((batch_size, prompt_seq_len, vocab_size))
    
    # For generated tokens
    for b in range(batch_size):
        for s in range(gen_len):
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
    print(f"logits = {logits}")
    print(f"action_mask = {action_mask}")
    print(f"prompt_len = {prompt_len}")
    print(f"max_gen_len = {max_gen_len}")
    print(f"{entropies=}")
    print(f"{torch.log(torch.tensor(vocab_size, dtype=torch.float))=}")
    
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
    print(f"{entropies[1]=}")
    print(f"{expected_entropy=}")
    assert torch.isclose(entropies[1], expected_entropy, atol=1e-5)
