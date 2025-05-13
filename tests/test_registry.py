# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import catalogue
from llmfoundry.utils import registry_utils

from compose_rl import registry
from compose_rl.reward_learning.functional import OutputLengthReward


def test_expected_registries_exist():
    existing_registries = {
        name for name in dir(registry)
        if isinstance(getattr(registry, name), registry_utils.TypedRegistry)
    }
    expected_registry_names = {
        'rewards',
        'kl_controllers',
    }

    assert existing_registries == expected_registry_names


def test_registry_init_code(tmp_path: pathlib.Path):
    register_code = """
from compose_rl.registry import rewards
from compose_rl.reward_learning.functional import OutputLengthReward

@rewards.register('test_reward')
class TestReward(OutputLengthReward):
    pass
import os
os.environ['TEST_ENVIRON_REGISTRY_KEY'] = 'test'
"""

    with open(tmp_path / 'init_code.py', 'w') as _f:
        _f.write(register_code)

    registry_utils.import_file(tmp_path / 'init_code.py')
    assert issubclass(registry.rewards.get('test_reward'), OutputLengthReward)
    del catalogue.REGISTRY[('llmfoundry', 'rewards', 'test_reward')]
    assert 'test_reward' not in registry.rewards
