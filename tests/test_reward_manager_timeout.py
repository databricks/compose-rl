# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RewardManager async timeout handling.

This test suite validates that the RewardManager properly handles async reward
computation timeouts by creating zero reward tensors as fallbacks. The tests
ensure that:

1. When an async reward computation times out, a zero reward tensor is created
2. The zero reward tensor has the correct shape, device, and dtype
3. The timeout is properly logged as an error
4. Multiple rewards (some timing out, some succeeding) are handled correctly
5. The make_zero_reward helper method works correctly
"""

import multiprocessing
from multiprocessing.pool import AsyncResult
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest
import torch
from composer.core import Precision
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from compose_rl.ppo.reward_manager import RewardManager, RewardOutput
from compose_rl.reward_learning import InferenceRewardModel


class MockAsyncResult(AsyncResult):
    """Mock AsyncResult that simulates timeout."""

    def __init__(
        self,
        should_timeout: bool = True,
        return_value: Optional[torch.Tensor] = None,
    ):
        # Don't call super().__init__ to avoid complex initialization
        self.should_timeout = should_timeout
        self.return_value = return_value

    def get(self, timeout: Optional[float] = None) -> torch.Tensor:
        if self.should_timeout:
            raise multiprocessing.TimeoutError('Mock timeout')
        if self.return_value is None:
            # Return a default tensor if none provided
            return torch.zeros(2, 10)
        return self.return_value


class MockRewardModel(InferenceRewardModel):
    """Mock reward model with timeout configuration."""

    BLOCKING_TIMEOUT = 1.0  # 1 second timeout for testing

    def __init__(self) -> None:
        # Don't call super().__init__ to avoid complex initialization
        pass

    def __call__(self, batch: Any) -> torch.Tensor:
        # Mock reward model computation
        batch_size = batch['zero_rewards'].shape[0]
        seq_len = batch['zero_rewards'].shape[1]
        return torch.zeros(batch_size, seq_len)


@pytest.fixture
def mock_reward_manager(
    tiny_gpt2_tokenizer: PreTrainedTokenizerBase,
) -> RewardManager:
    """Create a minimal RewardManager for testing."""
    config = DictConfig({
        'test_model': {
            'reward_type': 'inference_reward_model',
            'reward_coefficient': 1.0,
            'granularity': 'document',
        },
    })

    ref_config = {
        'model_config': {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'gpt2',
            'pretrained': True,
        },
    }

    # Create RewardManager with minimal initialization
    with patch.object(RewardManager, 'initialize_composer_model') as mock_init:
        with patch('compose_rl.ppo.reward_manager.spacy.load') as mock_spacy:
            mock_ref_model = Mock()
            mock_init.return_value = mock_ref_model

            # Mock spacy parser
            mock_parser = Mock()
            mock_spacy.return_value = mock_parser

            # Patch the reward registry and build_reward
            with patch(
                'compose_rl.ppo.reward_manager.rewards_registry',
            ) as mock_registry:
                with patch(
                    'compose_rl.ppo.reward_manager.build_reward',
                ) as mock_build:
                    mock_registry.get.return_value = InferenceRewardModel
                    mock_reward_model = MockRewardModel()
                    mock_build.return_value = mock_reward_model

                    reward_manager = RewardManager(
                        config=config,
                        ref_config=ref_config,
                        tokenizer=tiny_gpt2_tokenizer,  # type: ignore
                        max_seq_len=32,
                        fsdp_config=None,
                        precision=Precision.FP32,
                    )

                    # Manually set up the reward model
                    reward_manager.all_rewards['test_model'] = mock_reward_model
                    reward_manager.inference_rewards = ['test_model']

                    return reward_manager


def test_async_timeout_creates_zero_reward(
    mock_reward_manager: RewardManager,
) -> None:
    """Test that async timeout creates proper zero reward tensor."""
    batch_size, seq_len = 2, 10

    # Create test inputs
    action_mask = torch.ones(batch_size, seq_len)
    ref_output = (
        torch.zeros(batch_size, seq_len),  # ref_kl
        torch.zeros(batch_size, seq_len),  # ref_log_probs
    )

    # Create reward output with timeout AsyncResult
    timeout_async_result = MockAsyncResult(should_timeout=True)
    reward_output: RewardOutput = {
        'test_model': timeout_async_result,
    }

    # Mock KL controller
    mock_kl_ctl = Mock()
    mock_kl_ctl.value = 0.1

    # Test resolve_outputs with timeout
    with patch('compose_rl.ppo.reward_manager.log') as mock_log:
        outputs = mock_reward_manager.resolve_outputs(
            ref_output=ref_output,
            reward_output=reward_output,
            kl_ctl=mock_kl_ctl,
            action_mask=action_mask,
        )

    # Verify error was logged
    mock_log.error.assert_called_once()
    error_msg = mock_log.error.call_args[0][0]
    assert 'Timeout while waiting for test_model reward to finish' in error_msg
    assert 'Using a default reward of 0' in error_msg

    # Verify outputs have correct structure
    assert 'rewards' in outputs
    assert 'env_rewards' in outputs
    assert 'test_model_reward' in outputs  # reward name gets _reward suffix

    # Verify the reward tensor has correct shape and is all zeros
    reward_tensor = outputs['test_model_reward']
    assert reward_tensor.shape == (batch_size, seq_len)
    assert torch.allclose(reward_tensor, torch.zeros_like(reward_tensor))

    # Verify tensor properties match action_mask
    assert reward_tensor.device == action_mask.device
    assert reward_tensor.dtype == action_mask.dtype


def test_async_success_works_normally(
    mock_reward_manager: RewardManager,
) -> None:
    """Test that non-timeout case works normally."""
    batch_size, seq_len = 2, 10

    # Create test inputs
    action_mask = torch.ones(batch_size, seq_len)
    ref_output = (
        torch.zeros(batch_size, seq_len),  # ref_kl
        torch.zeros(batch_size, seq_len),  # ref_log_probs
    )

    # Create successful reward tensor
    expected_reward = torch.ones(batch_size, seq_len) * 0.5
    success_async_result = MockAsyncResult(
        should_timeout=False,
        return_value=expected_reward,
    )
    reward_output: RewardOutput = {
        'test_model': success_async_result,
    }

    # Mock KL controller
    mock_kl_ctl = Mock()
    mock_kl_ctl.value = 0.1

    # Test resolve_outputs without timeout
    outputs = mock_reward_manager.resolve_outputs(
        ref_output=ref_output,
        reward_output=reward_output,
        kl_ctl=mock_kl_ctl,
        action_mask=action_mask,
    )

    # Verify the reward tensor matches expected values
    reward_tensor = outputs['test_model_reward']
    assert reward_tensor.shape == (batch_size, seq_len)
    assert torch.allclose(reward_tensor, expected_reward * action_mask)


def test_mixed_timeout_and_success(mock_reward_manager: RewardManager) -> None:
    """Test scenario with multiple rewards where some timeout and some succeed.

    This test validates that the RewardManager can handle mixed scenarios where
    some async rewards timeout while others succeed normally.
    """
    batch_size, seq_len = 2, 10

    # Add another reward model to the manager
    mock_reward_manager.all_rewards['success_model'] = MockRewardModel()
    mock_reward_manager.reward_coefficients['success_model'] = 1.0

    # Create test inputs
    action_mask = torch.ones(batch_size, seq_len)
    ref_output = (
        torch.zeros(batch_size, seq_len),  # ref_kl
        torch.zeros(batch_size, seq_len),  # ref_log_probs
    )

    # Create mixed reward outputs
    expected_success_reward = torch.ones(batch_size, seq_len) * 0.7
    reward_output: RewardOutput = {
        'test_model': MockAsyncResult(should_timeout=True),  # This times out
        'success_model':
            expected_success_reward,  # This succeeds (direct tensor)
    }

    # Mock KL controller
    mock_kl_ctl = Mock()
    mock_kl_ctl.value = 0.1

    # Test resolve_outputs
    with patch('compose_rl.ppo.reward_manager.log'):
        outputs = mock_reward_manager.resolve_outputs(
            ref_output=ref_output,
            reward_output=reward_output,
            kl_ctl=mock_kl_ctl,
            action_mask=action_mask,
        )

    # Verify both rewards are in outputs
    assert 'test_model_reward' in outputs
    assert 'success_model_reward' in outputs

    # Verify timeout reward is zeros
    timeout_reward = outputs['test_model_reward']
    assert torch.allclose(timeout_reward, torch.zeros_like(timeout_reward))

    # Verify success reward has expected values
    success_reward = outputs['success_model_reward']
    assert torch.allclose(success_reward, expected_success_reward * action_mask)


def test_make_zero_reward_method() -> None:
    """Test the make_zero_reward static method directly."""
    # Test with various tensor shapes and types
    test_cases = [
        torch.ones(2, 10, dtype=torch.float32),
        torch.ones(3, 5, dtype=torch.float16),
        torch.ones(1, 15, dtype=torch.int64),
    ]

    for ref_tensor in test_cases:
        zero_reward = RewardManager.make_zero_reward(ref_tensor)

        # Verify shape, device, and dtype match
        assert zero_reward.shape == ref_tensor.shape
        assert zero_reward.device == ref_tensor.device
        assert zero_reward.dtype == ref_tensor.dtype

        # Verify all values are zero
        assert torch.allclose(zero_reward, torch.zeros_like(ref_tensor))


def test_timeout_error_import() -> None:
    """Test that TimeoutError is properly imported from multiprocessing."""
    # This verifies the import fix in the reward_manager.py file
    from compose_rl.ppo.reward_manager import TimeoutError
    assert TimeoutError is multiprocessing.TimeoutError
