# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

try:
    import llmfoundry
    del llmfoundry
except ImportError:
    raise ImportError(
        'The plugins repo requires llmfoundry package to be installed.' +
        'When installing plugins, please use one of the extras depending on which version of llmfoundry you are using.',
    )

import compose_rl.dpo as dpo
import compose_rl.reward_learning as reward_learning
from compose_rl import data, metrics, utils

__all__ = [
    'utils',
    'data',
    'dpo',
    'reward_learning',
    'metrics',
]
