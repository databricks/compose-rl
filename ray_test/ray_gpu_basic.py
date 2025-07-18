#!/usr/bin/env python3

# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""
Ray GPU Management - Basic Example

This is the simplest possible example of using Ray to manage GPU workloads.
Perfect for someone new to Ray who wants to understand the core concepts.
"""

import os
import time

import ray
import torch


@ray.remote(num_gpus=1)
def simple_gpu_task(task_id: int):
    """A minimal GPU task that just creates a tensor and does basic operations."""

    # Ray automatically manages which GPU this task gets
    gpu_ids = ray.get_gpu_ids()
    print(f"Task {task_id}: Using GPU {gpu_ids[0]}")

    # Create a tensor on the GPU
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)

    # Do some computation
    for i in range(3):
        x = x * 2
        time.sleep(0.5)  # Simulate work
        print(f"  Step {i+1}: tensor shape {x.shape}")

    return f"Task {task_id} completed on GPU {gpu_ids[0]}"


if __name__ == '__main__':
    # print current pic
    print(f"Current process ID: {os.getpid()}")

    # Initialize Ray
    ray.init()

    print('Available resources:', ray.cluster_resources())

    # Launch 2 tasks (one per GPU)
    tasks = [simple_gpu_task.remote(i) for i in range(2)]

    # Wait for results
    results = ray.get(tasks)

    print('Results:', results)
    ray.shutdown()
