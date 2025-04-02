# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Functional reward implementations."""

import logging
from typing import Any, MutableMapping

import torch

log = logging.getLogger(__name__)

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

    def validate_config(self):
        # There are no config requirements for this reward
        return

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

    def validate_config(self):
        # There are no config requirements for this reward
        return

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


class Gsm8kVerificationReward(Reward):

    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

    def validate_config(self):
        # There are no config requirements for this reward
        return

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Apply the reward for verifying the correct response (answer) from the model.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the verification reward.

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

        batch_size = rewards.shape[0]
        all_generated_texts = [x[1] for x in raw_untokenized_texts]
        for i in range(batch_size):
            _reward = self._score_generations(all_generated_texts[i], verified_answers[i])
            rewards[i, generated_lens[i] - 1] += _reward
        return rewards

    def _score_generations(self, prediction: str, label: int):
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        extracted = numbers[-1] if numbers else response
        return float(str(extracted).lower() == str(label).lower())


class Gsm8kFormatVerificationReward(Reward):

    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

    def validate_config(self):
        # There are no config requirements for this reward
        return

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Apply the reward for verifying the correct response format from the model.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the verification reward.

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
        for i in range(batch_size):
            _reward = self._score_generations(all_generated_texts[i])
            rewards[i, generated_lens[i] - 1] += _reward
        return rewards

    def _score_generations(self, prediction: str):
        try:
            # Split by newline and get the last element
            lines = answer.split('\n')
            if not lines:
                return 0.0

            # Use regex to match 4 #s followed by a space and capture everything after
            last_line = lines[-1]
            match = re.match(r'^#{4}\s+(.*)', last_line)
            return 1.0 if match else 0.0
        except Exception as e:
            return 0.0
