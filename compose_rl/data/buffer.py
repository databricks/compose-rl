# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """A dummy dataset class.

    This class is needed because how we interact with the dataset state in composer here:
    https://github.com/mosaicml/composer/blob/v0.16.4/composer/core/state.py#L579

    where it requires a DataLoader, so we are spoofing the `Dataset` part of the DataLoader here.
    """

    def __init__(self):
        self._state_dict = None

    def set_state_dict(self, state_dict: dict[str, Any]):
        """Sets the state dict.

        Args:
            state_dict (dict[str, Any]): the state dict of the dataloader we want to set.
        """
        self._state_dict = state_dict

    # (#146): Needs to be fixed for resumption to work
    def state_dict(self, num_samples: int, from_beginning: bool):
        """A dict containing the training state."""
        del num_samples, from_beginning
        return self._state_dict

    def __len__(self):
        # This is used for the dummy Distributed Sampler.
        return 1


class MinibatchRolloutBuffer(DataLoader):
    """A rollout buffer that operates on minibatches.

    This class is intended to wrap a pre-prepared minibatch buffer
    to iterate through it when composer's .fit() is called.
    Right now, we don't implement sampling from the buffer for simplicity.

    Args:
        buffer_size (int): the size of the buffer.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        # We need these here because of the PyTorch DataLoader.
        # This is noted in the class comments of `DummyDataset`.
        self.dataset = DummyDataset()
        self.dummy_distributed_sampler = DistributedSampler(self.dataset)

        super().__init__(self.dataset, sampler=self.dummy_distributed_sampler)

        self.max_buffer_size = cfg.get('max_buffer_size')
        self.index = 0

        self.state_dict = None
        self.buffer = []

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def reset(self) -> None:
        """Reset the buffer."""
        self.index = 0
        self.buffer = []

    def add(self, minibatch: dict[str, torch.Tensor]):
        """Adds a minibatch to the buffer.

        Args:
            minibatch (dict[str, torch.Tensor]): the minibatch to add to the buffer.
        """
        self.buffer.append(minibatch)
        if self.__len__() > self.max_buffer_size:
            warnings.warn(
                f'The buffer is now of size {self.__len__()} ' +
                'which is greater than the maximum size specified at initialization.',
            )

    def get_samples(self, num_samples: int) -> dict[str, torch.Tensor]:
        """Gets the next sample from the replay buffer."""
        raise NotImplementedError(
            f'sample is not implemented for MinibatchRolloutBuffer.',
        )

    def _get_samples_from_indices(
        self,
        batch_inds: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Get samples from certain indices from the MinibatchRolloutBuffer.

        Args:
            batch_inds (np.ndarray): a list of indices for which batches to retrieve.
        """
        raise NotImplementedError(
            f'_get_samples is not implemented for MinibatchRolloutBuffer.',
        )

    def set_state_dict(self, state_dict: dict[str, Any], epoch: int):
        """Sets the state dict from the replay buffer.

        This is needed because we're using .fit() in composer, which calls
        state_dict to save the state.

        Args:
            state_dict (dict[str, Any]): the state dict we are trying to set our buffer's state dict
            epoch (int): the true epoch number in the prompt dataloader, which is different from the
                epoch number in composer. This is because composer considers one `epoch` to iterate through
                the dataset in `.fit` once.
        """
        state_dict['epoch'] = epoch
        log.info(f'Saving state dict to: {state_dict}')
        self.dataset.set_state_dict(  # pyright: ignore[reportGeneralTypeIssues]
            state_dict)
