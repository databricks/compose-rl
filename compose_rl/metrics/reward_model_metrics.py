# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
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
        self.correct += (batch['chosen_scores'] >
                         batch['rejected_scores']).sum().detach().cpu()

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total
