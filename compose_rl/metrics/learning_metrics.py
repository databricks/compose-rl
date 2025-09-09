# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torchmetrics import Metric


class TestLossMetric(Metric):
    """Training loss metric for tracking loss components.
    
    Tracks different components of training loss including total loss,
    implicit rewards, KL divergences, estimated reward, and sequence entropies.
    """
    
    # Make torchmetrics call update only once
    full_state_update = False
    
    def __init__(self, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Track total loss
        self.add_state(
            'total_loss_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        
        # Track implicit rewards loss
        self.add_state(
            'implicit_rewards_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        
        # Track reverse KL divergence loss
        self.add_state(
            'reverse_kl_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        
        # Track forward KL divergence loss
        self.add_state(
            'forward_kl_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        
        # Track estimated reward loss
        self.add_state(
            'estimated_reward_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        
        # Track sequence entropies loss
        self.add_state(
            'sequence_entropies_sum',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        
        # Track number of samples for averaging
        self.add_state(
            'count',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
    
    def update(self, batch: dict, output_logits: torch.Tensor):
        """Update metric state with loss components.
        
        Args:
            batch: Dictionary containing loss components with keys:
                - 'total_loss': Total training loss
                - 'implicit_rewards_loss': Implicit rewards loss component
                - 'reverse_kl_loss': Reverse KL divergence loss
                - 'forward_kl_loss': Forward KL divergence loss  
                - 'estimated_reward_loss': Estimated reward loss
                - 'sequence_entropies_loss': Sequence entropies loss
        """
        
        batch_size = 1  # Assuming loss values are already averaged over batch
        
        # Update loss sums
        if 'total_loss' in batch:
            self.total_loss_sum += batch['total_loss'].detach().cpu()
            
        if 'implicit_rewards_loss' in batch:
            self.implicit_rewards_sum += batch['implicit_rewards_loss'].detach().cpu()
            
        if 'reverse_kl_loss' in batch:
            self.reverse_kl_sum += batch['reverse_kl_loss'].detach().cpu()
            
        if 'forward_kl_loss' in batch:
            self.forward_kl_sum += batch['forward_kl_loss'].detach().cpu()
            
        if 'estimated_reward_loss' in batch:
            self.estimated_reward_sum += batch['estimated_reward_loss'].detach().cpu()
            
        if 'sequence_entropies_loss' in batch:
            self.sequence_entropies_sum += batch['sequence_entropies_loss'].detach().cpu()
        
        self.count += batch_size
    
    def compute(self):
        """Compute average losses."""
        if self.count == 0:
            return {
                'total': torch.tensor(0.0),
                'implicit_rewards': torch.tensor(0.0),
                'reverse_kl': torch.tensor(0.0),
                'forward_kl': torch.tensor(0.0),
                'estimated_reward': torch.tensor(0.0),
                'sequence_entropies': torch.tensor(0.0),
            }
        
        return {
            'total': self.total_loss_sum / self.count,
            'implicit_rewards': self.implicit_rewards_sum / self.count,
            'reverse_kl': self.reverse_kl_sum / self.count,
            'forward_kl': self.forward_kl_sum / self.count,
            'estimated_reward': self.estimated_reward_sum / self.count,
            'sequence_entropies': self.sequence_entropies_sum / self.count,
        }


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
        del output_logits
        if 'total_loss' in batch:
            self.loss_sum += batch['total_loss'].detach().cpu()
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
        del output_logits
        if 'implicit_rewards_loss' in batch:
            self.loss_sum += batch['implicit_rewards_loss'].detach().cpu()
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
        self.loss_key = f'{kl_type}_kl_loss'
        
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
        del output_logits
        if self.loss_key in batch:
            self.loss_sum += batch[self.loss_key].detach().cpu()
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
        del output_logits
        if 'estimated_reward_loss' in batch:
            self.loss_sum += batch['estimated_reward_loss'].detach().cpu()
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
        del output_logits
        if 'sequence_entropies_loss' in batch:
            self.loss_sum += batch['sequence_entropies_loss'].detach().cpu()
            self.count += 1
    
    def compute(self):
        return self.loss_sum / self.count if self.count > 0 else torch.tensor(0.0)
