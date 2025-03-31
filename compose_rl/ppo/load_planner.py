# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
from typing import Optional
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, create_default_local_save_plan, create_default_local_load_plan, SavePlan
import dataclasses

from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    Metadata
)
from torch.distributed.checkpoint._nested_dict import flatten_state_dict


log = logging.getLogger(__name__)

class PPOModelLoadPlanner(DefaultLoadPlanner):    
    def create_local_plan(self):

        self.metadata_has_critic_key = False
        for key, _ in self.metadata.state_dict_metadata.items():
            if 'critic_head' in key:
                self.metadata_has_critic_key = True

        self.state_dict = self.convert_state_dict(self.state_dict)
        plan = create_default_local_load_plan(self.state_dict, self.metadata)
        
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)

        self.plan = plan
        return self.plan

    def convert_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in self.state_dict.items():
            # If the metadata has a critic key, then we should assume we are 
            # trying to autoresume and not replace any keys. 
            # However, the other case is we want to load another model generated 
            # by LLM-foundry. The code below will properly remap keys
            # to ensure we can properly load. 
            if not self.metadata_has_critic_key and 'state.model.' in key:
                key = key.replace('lm_backbone.', '')

            new_state_dict[key] = value
        
        return new_state_dict
