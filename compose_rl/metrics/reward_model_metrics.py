# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


class PairwiseRewardClassificationAccuracy(Metric):
    """Pairwise reward classifcation accuracy.

    Computes the accuracy of a pairwise reward model, by the score of chosen
    being greater than the score of the rejected.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False, **kwargs: Any):
        # State from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'correct',
            default=torch.tensor(0.),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, batch: dict, output_logits: torch.Tensor):
        del output_logits
        bs, _ = batch['chosen_scores'].shape

        self.total += bs
        self.correct += (batch['chosen_scores']
                         > batch['rejected_scores']).sum().detach().cpu()

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class BinaryRewardClassificationAccuracy(Metric):
    """Classification accuracy metric.

    Computes the accuracy of a classifier by comparing predictions from logits
    against ground truth labels. Handles both binary and multi-class
    classification.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self,
        threshold: float = 0.5,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        """Initialize the metric.

        Args:
            binary: If True, treats as binary classification with sigmoid.
                   If False, treats as multi-class with softmax.
            threshold: Decision threshold for binary classification
            dist_sync_on_step: Synchronize metric state across processes
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold

        self.add_state(
            'correct',
            default=torch.tensor(0.),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, batch: dict, output_logits: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            batch: Dictionary containing 'output_scores' and 'labels' and 'attention_mask'
            output_logits: `None`
        """
        del output_logits
        logits = batch['output_scores']
        targets = batch['labels']
        prompt_lens = batch['prompt_len']
        targets_mask = torch.logical_not(targets == -100)  # 0 at places where target label = -100, 1 otherwise

        assert logits.shape[0] == targets.shape[0] #'Batch sizes must match'
        assert logits.shape[1] == targets.shape[1] # seq length should match

        # TODO (raj): Handle multi-class classification with logging
        n_class = logits.size(-1)
        if n_class >=2: 
            predictions = torch.argmax(logits, dim = -1).detach()
        else:
            probs = torch.sigmoid(logits.squeeze())
            predictions = (probs > self.threshold).long()

        self.correct += ((predictions == targets) * targets_mask).sum().detach().cpu()
        self.total += torch.sum(targets_mask)

    def compute(self):
        """Compute the accuracy."""
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total
