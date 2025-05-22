# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

# # Copyright 2024 MosaicML ComposeRL authors
# # SPDX-License-Identifier: Apache-2.0

# """Tests for the MATHVerifierReward class."""

# from typing import Any

# import pytest
# import torch
# from transformers import AutoTokenizer

# from compose_rl.reward_learning import MATHVerifierReward

# @pytest.fixture
# def reward() -> MATHVerifierReward:
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#     return MATHVerifierReward(reward=10.0, tokenizer=tokenizer)

# def test_call_base_verifer_invalid_input(reward: MATHVerifierReward) -> None:
#     invalid_batch: dict[str, torch.Tensor] = {
#         'zero_rewards': torch.zeros((2, 6)),
#         'generated_lens': torch.tensor([6, 6]),
#     }
#     with pytest.raises(AssertionError):
#         reward(invalid_batch)

# @pytest.mark.parametrize(
#     'batch, expected_rewards',
#     [
#         (
#             {
#                 'zero_rewards': torch.zeros((3, 5)),
#                 'raw_untokenized_texts': [
#                     ('', 'Answer is \\boxed{24}'),
#                     (
#                         '',
#                         'We see that $f(y) + g(y) = y^4 -3y^3+y-3 +y^3+7y^2-2.$ Simplifying, we get $\\boxed{y^4-2y^3+7y^2+y-5}$.',
#                     ),
#                     (
#                         '',
#                         'Setting the exponents equal, we get that $36n=72$, or $n=\frac{72}{36}=\\boxed{2}$.',
#                     ),
#                 ],
#                 'verified_answers': ['24', 'y^4-2y^3+7y^2+y-4', '2'],
#                 'generated_lens': torch.tensor([5, 5, 3]),
#             },
#             [(0, 4, 10.0), (1, 4, 0.0), (2, 2, 10.0)],
#         ),
#         (
#             {
#                 'zero_rewards': torch.zeros((2, 6)),
#                 'raw_untokenized_texts': [
#                     ('', '\\boxed{\frac{1}{2}}'),
#                     ('', '\\boxed{45}'),
#                 ],
#                 'verified_answers': ['\frac{1}{2}', '4'],
#                 'generated_lens': torch.tensor([6, 6]),
#             },
#             [(0, 5, 10.0), (1, 5, 0.0)],
#         ),
#     ],
# )
# def test_math_verifier(
#     reward: MATHVerifierReward,
#     batch: dict[str, Any],
#     expected_rewards: list[tuple[int, int, float]],
# ) -> None:
#     result = reward(batch)
#     assert result.shape == batch['zero_rewards'].shape
#     for idx, pos, expected in expected_rewards:
#         assert pytest.approx(result[idx, pos].item(), abs=1e-6) == expected
