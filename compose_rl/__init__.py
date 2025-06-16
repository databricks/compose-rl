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

from compose_rl import algorithms, data, metrics, utils

__all__ = [
    'algorithms',
    'utils',
    'data',
    'metrics',
]
