# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torchmetrics import Metric


class TestTotalLossMetric(Metric):
    """Metric for tracking total training loss."""
    
    full_state_update = False
    
    def __init__(self, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'loss_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'count',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
    
    def update(self, batch: dict, output_logits: torch.Tensor):
        
        if 'total' in batch:
            self.loss_sum += batch['total'].detach().cpu().item()
            self.count += 1
    
    def compute(self):
        return self.loss_sum / self.count if self.count > 0 else torch.tensor(0.0)


class TestImplicitRewardsLossMetric(Metric):
    """Metric for tracking implicit rewards loss."""
    
    full_state_update = False
    
    def __init__(self, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'loss_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'count',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
    
    def update(self, batch: dict, output_logits: torch.Tensor):
        
        if 'implicit_rewards' in batch:
            self.loss_sum += batch['implicit_rewards'].detach().cpu().item()
            self.count += 1
    
    def compute(self):
        return self.loss_sum / self.count if self.count > 0 else torch.tensor(0.0)


class TestKLDivergenceLossMetric(Metric):
    """Metric for tracking KL divergence losses."""
    
    full_state_update = False
    
    def __init__(self, kl_type: str = 'reverse', dist_sync_on_step: bool = False, **kwargs: Any):
        """Initialize KL divergence loss metric.
        
        Args:
            kl_type: Type of KL divergence to track ('reverse' or 'forward')
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.kl_type = kl_type
        self.loss_key = f'{kl_type}_kl'
        
        self.add_state(
            'loss_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'count',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
    
    def update(self, batch: dict, output_logits: torch.Tensor):
        
        if self.loss_key in batch:
            self.loss_sum += batch[self.loss_key].detach().cpu().item()
            self.count += 1
    
    def compute(self):
        return self.loss_sum / self.count if self.count > 0 else torch.tensor(0.0)


class TestEstimatedRewardLossMetric(Metric):
    """Metric for tracking estimated reward loss."""
    
    full_state_update = False
    
    def __init__(self, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'loss_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'count',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
    
    def update(self, batch: dict, output_logits: torch.Tensor):
        # print("keys in batch: ", batch.keys())
        # print("updating in estimated reward loss metric")
        if 'estimated_reward' in batch:
            self.loss_sum += batch['estimated_reward'].detach().cpu().item()
            self.count += 1
    
    def compute(self):
        return self.loss_sum / self.count if self.count > 0 else torch.tensor(0.0)


class TestSequenceEntropiesLossMetric(Metric):
    """Metric for tracking sequence entropies loss."""
    
    full_state_update = False
    
    def __init__(self, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'loss_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'count',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
    
    def update(self, batch: dict, output_logits: torch.Tensor):
        
        if 'sequence_entropies' in batch:
            self.loss_sum += batch['sequence_entropies'].detach().cpu().item()
            self.count += 1
    
    def compute(self):
        return self.loss_sum / self.count if self.count > 0 else torch.tensor(0.0)