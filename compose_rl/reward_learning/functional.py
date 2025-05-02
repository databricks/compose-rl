# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Functional reward implementations."""

import logging
import re
from abc import abstractmethod
from typing import Any, MutableMapping

import torch

log = logging.getLogger(__name__)

from compose_rl.data.rlvr_utils import (
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from compose_rl.reward_learning.base_reward import Reward, Tokenizer


class IncreasingNumbersReward(Reward):

    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

    @staticmethod
    def is_number(text: str):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Creates a reward based on the number of generated increasing numbers.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the increasing numbers reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'raw_untokenized_texts' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        raw_untokenized_texts = batch['raw_untokenized_texts']
        generated_lens = batch['generated_lens']

        batch_size = rewards.shape[0]
        all_generated_texts = [x[1] for x in raw_untokenized_texts]
        curr_rewards = []
        for gen_text in all_generated_texts:
            gen_tokens = gen_text.split()
            number_tokens = [
                float(token)
                for token in gen_tokens
                if IncreasingNumbersReward.is_number(token)
            ]
            if len(number_tokens) > 0:
                sorted_count = 1
                previous_token = number_tokens[0]
                for token in number_tokens[1:]:
                    if token > previous_token:
                        sorted_count += 1
                        previous_token = token
                    else:
                        break
                curr_rewards.append((sorted_count) / max(len(gen_tokens), 1))
            else:
                curr_rewards.append(0)
        curr_rewards = torch.tensor(curr_rewards).to(
            rewards.device,
        ).type(rewards.dtype)
        rewards[torch.arange(batch_size), generated_lens - 1] += curr_rewards
        return rewards


class ShortResponseReward(Reward):

    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

        self.reward = cfg['reward']
        self.len_threshold = cfg['len_threshold']
        log.info(
            f'Adding a reward of {self.reward} if a model generates ' +
            f'tokens under the length {self.len_threshold}',
        )

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Apply the reward to the EOS tokens and nothing else.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the short response reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        generated_lens = batch['generated_lens']
        bs = generated_lens.size(0)
        for i in range(bs):
            if generated_lens[i] <= self.len_threshold:
                rewards[i, generated_lens[i] - 1] += self.reward
        return rewards


class BadGenerationEndReward(Reward):

    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

        self.reward = cfg.get('reward', None)
        self.eos_penalty = cfg.get('eos_penalty', None)

        assert self.reward is not None, 'reward must be provided in the config'
        assert self.eos_penalty is not None, 'eos_penalty must be provided in the config'

        # Extra special tokens for any other formats with pseudo EOS alternatives like ChatML
        self.extra_special_tokens = [
            str(tok) for tok in cfg.get('extra_special_tokens', [])
        ]
        self.extra_special_token_ids = []
        if self.extra_special_tokens != []:
            self.extra_special_token_ids.extend([
                tok[0] for tok in self.tokenizer(
                    self.extra_special_tokens,
                )  # pyright: ignore
                ['input_ids']
            ])
        if self.eos_penalty:
            # Because tokenizer can be optional, we need to ignore
            self.extra_special_token_ids.append(
                self.tokenizer.eos_token_id,  # pyright: ignore
            )
        log.info(
            f'Subtracting a reward of {self.reward} if a model does not' +
            f'end with an EOS or given set of special tokens',
        )

    def validate_config(self):
        if 'eos_penalty' not in self.cfg:
            raise KeyError(
                f'Required field eos_penalty is missing from BadGenerationEndReward config',
            )
        if 'reward' not in self.cfg:
            raise KeyError(
                f'Required field reward is missing from BadGenerationEndReward config',
            )

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Rewards if the generated sequences don't end in EOS or special token.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the bad generation end reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'seq_lens' in batch.keys()
        assert 'input_ids' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        seq_lens = batch['seq_lens']
        input_ids = batch['input_ids']
        generated_lens = batch['generated_lens']

        for i in range(generated_lens.size(0)):
            curr_end_token_id = input_ids[i, seq_lens[i] - 1]
            if curr_end_token_id.item() not in self.extra_special_token_ids:
                rewards[i, generated_lens[i] - 1] += self.reward
        return rewards


class OutputLengthReward(Reward):

    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)
        self.max_gen_len = self.cfg.get('max_gen_len', None)
        assert self.max_gen_len is not None, 'max_gen_len must be provided in the config'

    def validate_config(self):
        if 'max_gen_len' not in self.cfg:
            raise KeyError(
                f'Required field max_gen_len is missing from OutputLengthReward config',
            )

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Rewards based on how many output tokens are generated.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the output length reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        generated_lens = batch['generated_lens']

        batch_size = rewards.shape[0]
        curr_rewards = generated_lens / self.max_gen_len
        rewards[torch.arange(batch_size), generated_lens - 1] += curr_rewards
        return rewards


class BaseVerifierReward(Reward):
    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)
        self.reward = cfg.get('reward', 1.0)
        log.info(
            f'Using reward value of {self.reward} for {self.__class__.__name__} verifier',
        )

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Apply the reward for verifying the correct answer from the model.

        Currently verifier rewards are only applied to the last token of the sequence.

        Args:
            batch (dict): The input batch containing all information needed.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'raw_untokenized_texts' in batch.keys()
        assert 'verified_answers' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        raw_untokenized_texts = batch['raw_untokenized_texts']
        verified_answers = batch['verified_answers']
        generated_lens = batch['generated_lens']
        metadata = batch.get('metadata', None)

        batch_size = rewards.shape[0]
        all_generated_texts = [x[1] for x in raw_untokenized_texts]
        for i in range(batch_size):
            # Process based on verifier type
            if self.needs_extraction():
                _answer = self.extract_solution(
                    all_generated_texts[i],
                    metadata[i] if metadata is not None else None,
                )
                _reward = self.score_generations(
                    _answer,
                    verified_answers[i],
                    metadata[i] if metadata is not None else None,
                )
            else:
                # Score directly without extraction
                _reward = self.score_generations(
                    all_generated_texts[i],
                    verified_answers[i],
                    metadata[i] if metadata is not None else None,
                )

            rewards[i, generated_lens[i] - 1] += _reward
        return rewards

    def needs_extraction(self) -> bool:
        """Determine if this verifier needs to extract solutions before scoring.

        Override in child classes if needed.

        Returns:
            bool: True if extraction is needed, False otherwise.
        """
        return True

    def extract_solution(self, text: str, metadata: str | None = None) -> str:
        """Extract the solution from text.

        Default implementation raises error; override in child classes if needed.

        Args:
            text (str): The generated text.
            metadata (str | None): The metadata associated with the sample. Defaults to `None`.

        Returns:
            str: The extracted solution.
        """
        raise NotImplementedError(
            'Subclasses must implement `extract_solution` if `needs_extraction` returns True.',
        )

    @abstractmethod
    def score_generations(
        self,
        answer: str,
        label: str,
        metadata: str | None = None,
    ) -> float:
        """Score the generated answer against the label.

        Default implementation raises error; override in child classes if needed.

        Args:
            answer (str): The extracted answer.
            label (str): The verified answer.
            metadata (str | None): The metadata associated with the sample. Defaults to `None`.

        Returns:
            float: The reward score.
        """
        raise NotImplementedError(
            'Subclasses must implement `score_generations` definition.',
        )


class GSM8KAnswerVeriferReward(BaseVerifierReward):

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

    def extract_solution(self, text: str, metadata: str | None = None) -> str:
        """Extract numerical solution from GSM8K-style responses."""
        del metadata  # unused

        numbers = re.findall(r'-?[\d,]*\.?\d+', text)
        final_answer = ''
        if len(numbers) > 0:
            final_answer = numbers[-1].strip().lower().replace(',', '').replace(
                '$',
                '',
            )

        return final_answer

    def score_generations(
        self,
        answer: str,
        label: str,
        metadata: str | None = None,
    ) -> float:
        """Score based on exact match."""
        del metadata  # unused

        return self.reward if answer == label else 0.0


class GSM8KFormatVeriferReward(BaseVerifierReward):

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

    def needs_extraction(self) -> bool:
        """Indicate that this verifier doesn't need extraction."""
        return False

    def score_generations(
        self,
        answer: str,
        label: str,
        metadata: str | None = None,
    ) -> float:
        """Check if answer follows the expected format with '####' marker."""
        del label, metadata  # unused

        solution = re.search(r'####.*?([\d,]+(?:\.\d+)?)', answer)
        return self.reward if solution is not None else 0.0


class MATHVerifierReward(BaseVerifierReward):

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

    def extract_solution(self, text: str, metadata: str | None = None) -> str:
        """Extract solution from MATH responses."""
        del metadata  # unused

        last_boxed_string = last_boxed_only_string(text)
        if not last_boxed_string:
            # No boxed string found, so we can't evaluate
            return ''

        unnormalized_answer = remove_boxed(last_boxed_string)
        final_answer = normalize_final_answer(unnormalized_answer)
        return final_answer

    def score_generations(
        self,
        answer: str,
        label: str,
        metadata: str | None = None,
    ) -> float:
        """Score based on exact match."""
        del metadata  # unused

        if answer.strip() == label.strip() or is_equiv(answer, label):
            return self.reward
        return 0.0
