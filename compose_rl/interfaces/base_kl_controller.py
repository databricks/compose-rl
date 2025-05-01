# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import torch

__all__ = ['BaseKLController']
class BaseKLController():
    """Base KL Controller class."""

    def __init__(self, kl_config: dict, device: str):
        self.device = device

    @abstractmethod
    def update(self, current: torch.Tensor, n_steps: int):
        """Updates the KL coefficient.

        Args:
            current (torch.Tensor): Current KL Divergence
            n_steps (int): Number of steps taken

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self):
        """Returns scalar KL coefficient value."""
        raise NotImplementedError

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """Loads the state dict of the KL controller if necessary."""
        return
