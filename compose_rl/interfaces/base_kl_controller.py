# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

import torch

__all__ = ['BaseKLController']


class BaseKLController(ABC):
    """Abstract base class for KL controller classes.

    Args:
        device: the device to run the KL controller on.
        *args: additional arguments.
        **kwargs: additional keyword arguments.
    """

    def __init__(self, device: str, *args: Any, **kwargs: Any):
        del device, args, kwargs  # unsued here
        pass

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
