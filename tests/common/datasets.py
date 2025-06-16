# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import Dataset


class PairwisePreference(Dataset):
    """A dataset of numbers where the output is the parity.

    Args:
        size (int): Number of samples (default: 100)
    """

    def __init__(
        self,
        size: int = 8,
        prompt_len: int = 5,
        max_seq_len: int = 10,
    ):
        self.size = size
        self.prompt_len = prompt_len
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Return samples that are randint 64 with seq len self.sequenece_length
        return {
            'chosen': torch.ones((self.max_seq_len,)).int(),
            'rejected': torch.zeros((self.max_seq_len,)).int(),
            'chosen_len': torch.Tensor([self.max_seq_len]).to(torch.int64),
            'rejected_len': torch.Tensor([self.max_seq_len]).to(torch.int64),
            'prompt_len': torch.Tensor([self.prompt_len]).to(torch.int64),
        }


class FineGrainedPreference(Dataset):
    """Mock dataset for testing fine-grained preferences."""

    def __init__(self, size: int = 8, max_seq_len: int = 10):
        self.max_seq_len = max_seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        text = torch.ones(self.max_seq_len, dtype=torch.int64)
        label = torch.zeros(1, dtype=torch.float32)
        text_len = torch.tensor([self.max_seq_len], dtype=torch.int64)

        return {
            'text': text,
            'labels': label,
            'text_len': text_len,
        }


class PromptDataset(Dataset):

    def __init__(self, size: int = 8, prompt_len: int = 5):
        self.size = size
        self.prompt_len = prompt_len

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return {
            'prompt': torch.ones((self.prompt_len,)).int(),
            'prompt_len': torch.Tensor([self.prompt_len]).to(torch.int64),
        }


class VerifiablePromptDataset(Dataset):

    def __init__(self, size: int = 8, prompt_len: int = 5):
        self.size = size
        self.prompt_len = prompt_len

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return {
            'prompt': torch.ones((self.prompt_len,)).int(),
            'prompt_len': torch.Tensor([self.prompt_len]).to(torch.int64),
            'verified_answer': '1',
        }


class VerifiableMessagesDataset(Dataset):

    def __init__(self, size: int = 8, prompt_len: int = 5):
        self.size = size
        self.prompt_len = prompt_len

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        messages = [{'role': 'user', 'content': 'F' * self.prompt_len}]  # bit of a hack, but it works
        mock_prompt = torch.ones((len(messages[0]['content']),)).int()
        return {
            'messages': messages,
            'prompt': mock_prompt,
            'prompt_len': torch.Tensor([len(messages[0]['content'])]).to(torch.int64),
            'verified_answer': 'Paris',
        }