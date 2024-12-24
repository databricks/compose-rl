# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from composer.utils import dist, get_device, reproducibility


# Copied from https://github.com/mosaicml/composer/blob/7656db4806ebd13f573210f7d3e1b5f26d7aed39/tests/fixtures/autouse_fixtures.py
@pytest.fixture(autouse=True)
def initialize_dist(request: pytest.FixtureRequest):
    """Initialize the default PyTorch distributed process group for tests."""
    gpu = request.node.get_closest_marker('gpu')
    dist.initialize_dist(get_device('gpu' if gpu is not None else 'cpu'))


@pytest.fixture(autouse=True)
def clear_cuda_cache(request: pytest.FixtureRequest):
    """Clear memory between GPU tests."""
    marker = request.node.get_closest_marker('gpu')
    if marker is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Only gc on GPU tests as it 2x slows down CPU tests


@pytest.fixture
def random_seed() -> int:
    return 17


@pytest.fixture(autouse=True)
def seed_all(random_seed: int):
    """Sets the seed for reproducibility."""
    reproducibility.seed_all(random_seed)
