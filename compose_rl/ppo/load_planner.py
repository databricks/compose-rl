# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from typing import Any

from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    create_default_local_load_plan,
)

log = logging.getLogger(__name__)


class PPOModelLoadPlanner(DefaultLoadPlanner):
    def create_local_plan(self):
        self.metadata_has_critic_key = False  # type: ignore
        for key in self.metadata.state_dict_metadata.keys():  # type: ignore
            if 'critic_head' in key:
                self.metadata_has_critic_key = True

        self.state_dict = self.convert_state_dict(self.state_dict)
        plan = create_default_local_load_plan(
            self.state_dict,
            self.metadata,  # type: ignore
        )

        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)

        self.plan = plan  # type: ignore
        return self.plan

    def convert_state_dict(self, state_dict: dict[str, Any]):
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
