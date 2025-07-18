#!/usr/bin/env python3

# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""
Ray GPU Scheduling Demo

This demonstrates how Ray schedules tasks based on GPU resource availability.
Key question: Can heavy_gpu_tasks (1.0 GPU) start when light_gpu_tasks (0.5 GPU each) are running?

Answer: NO - Ray waits until sufficient resources are available.
"""

import os
import time
from datetime import datetime

import ray
import torch


def timestamp():
    """Get current timestamp for logging."""
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


@ray.remote(num_gpus=0.5)
def light_gpu_task(task_id: int, duration: int = 10):
    """Light task that uses 0.5 GPU and runs for specified duration."""
    gpu_ids = ray.get_gpu_ids()
    pid = os.getpid()

    print(
        f"[{timestamp()}] 🟡 Light task {task_id} STARTED (PID: {pid}, GPU: {gpu_ids})"
    )

    # Create some GPU work
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)

    # Simulate work for the specified duration
    for i in range(duration):
        x = torch.mm(x, x.T)
        time.sleep(1)
        if i % 3 == 0:  # Progress update every 3 seconds
            print(
                f"[{timestamp()}] 🟡 Light task {task_id} working... ({i+1}/{duration}s)"
            )

    print(f"[{timestamp()}] 🟡 Light task {task_id} FINISHED")
    return f"Light task {task_id} completed"


@ray.remote(num_gpus=1.0)
def heavy_gpu_task(task_id: int, duration: int = 5):
    """Heavy task that needs full GPU and runs for specified duration."""
    gpu_ids = ray.get_gpu_ids()
    pid = os.getpid()

    print(
        f"[{timestamp()}] 🔴 Heavy task {task_id} STARTED (PID: {pid}, GPU: {gpu_ids[0]})"
    )

    # Create heavier GPU work
    device = torch.device('cuda')
    x = torch.randn(2000, 2000, device=device)

    # Simulate work
    for i in range(duration):
        x = torch.mm(x, x.T)
        time.sleep(1)
        print(
            f"[{timestamp()}] 🔴 Heavy task {task_id} working... ({i+1}/{duration}s)"
        )

    print(f"[{timestamp()}] 🔴 Heavy task {task_id} FINISHED")
    return f"Heavy task {task_id} completed"


@ray.remote
def resource_monitor():
    """Monitor available resources."""
    total = ray.cluster_resources()
    available = ray.available_resources()

    return {
        'timestamp': timestamp(),
        'total_gpus': total.get('GPU', 0),
        'available_gpus': available.get('GPU', 0),
        'available_cpus': available.get('CPU', 0),
    }


def demonstrate_scheduling():
    """Demonstrate Ray's scheduling behavior."""

    print('=' * 60)
    print('RAY GPU SCHEDULING DEMONSTRATION')
    print('=' * 60)
    print()

    ray.init()

    # Check initial resources
    initial_resources = ray.get(resource_monitor.remote())
    print(f"Initial resources: {initial_resources}")
    print()

    print(
        'SCENARIO: Testing if heavy tasks can start while light tasks are running'
    )
    print('- Light tasks: 0.5 GPU each, 10 seconds duration')
    print('- Heavy tasks: 1.0 GPU each, 5 seconds duration')
    print('- With 2 GPUs: 4 light tasks should fill both GPUs (2 per GPU)')
    print('- Heavy tasks should WAIT until light tasks finish')
    print()

    # Launch tasks in specific order to demonstrate scheduling
    print(
        f"[{timestamp()}] 🚀 Launching 4 light tasks (should fill both GPUs)..."
    )

    light_tasks = []
    for i in range(4):
        task = light_gpu_task.remote(i, duration=10)
        light_tasks.append(task)
        time.sleep(0.5)  # Small delay to see launch order

    # Wait a moment for light tasks to start
    time.sleep(2)

    # Check resources after light tasks start
    mid_resources = ray.get(resource_monitor.remote())
    print(f"[{timestamp()}] Resources after light tasks start: {mid_resources}")
    print()

    # Now launch heavy tasks - these should be QUEUED
    print(f"[{timestamp()}] 🚀 Launching 2 heavy tasks (should be QUEUED)...")

    heavy_tasks = []
    for i in range(2):
        task = heavy_gpu_task.remote(i, duration=5)
        heavy_tasks.append(task)
        time.sleep(0.5)

    print()
    print(
        '⏳ OBSERVATION: Heavy tasks will wait until sufficient GPU resources are free!'
    )
    print('   - Each light task uses 0.5 GPU')
    print('   - Each heavy task needs 1.0 GPU')
    print(
        '   - Heavy tasks must wait for 2 light tasks to finish to get 1.0 GPU'
    )
    print()

    # Monitor resources periodically
    for i in range(3):
        time.sleep(3)
        current_resources = ray.get(resource_monitor.remote())
        print(f"[{timestamp()}] Current resources: {current_resources}")

    # Wait for all tasks to complete
    print(f"\n[{timestamp()}] ⏳ Waiting for all tasks to complete...")

    light_results = ray.get(light_tasks)
    heavy_results = ray.get(heavy_tasks)

    print(f"\n[{timestamp()}] ✅ All tasks completed!")
    print('Light task results:', light_results)
    print('Heavy task results:', heavy_results)

    # Final resource check
    final_resources = ray.get(resource_monitor.remote())
    print(f"Final resources: {final_resources}")

    ray.shutdown()


def explain_scheduling():
    """Explain Ray's scheduling algorithm."""
    print('\n' + '=' * 60)
    print('RAY SCHEDULING EXPLAINED')
    print('=' * 60)
    print(
        """
Ray's resource scheduler works like this:

1. RESOURCE TRACKING:
   - Ray tracks total and available resources (GPUs, CPUs, memory)
   - Each task declares its resource requirements (@ray.remote(num_gpus=X))

2. TASK QUEUE:
   - Tasks are queued when submitted with .remote()
   - Ray maintains a queue of pending tasks

3. SCHEDULING DECISIONS:
   - Ray checks if enough resources are available for each queued task
   - Tasks only start when their FULL resource requirements can be met
   - No partial allocation - if task needs 1.0 GPU, it waits for 1.0 GPU

4. FRACTIONAL RESOURCES:
   - 0.5 GPU tasks: 2 can run on same physical GPU
   - 1.0 GPU tasks: Need exclusive access to 1 physical GPU
   - If 2×0.5 GPU tasks are running, 1.0 GPU task must WAIT

5. SCHEDULING ORDER:
   - Generally FIFO (first-in-first-out)
   - But resource availability affects actual execution order
   - Tasks with available resources start first

KEY INSIGHT:
Heavy tasks (1.0 GPU) CANNOT start while light tasks (0.5 GPU each)
occupy all GPU resources, even if the physical GPU isn't fully utilized.

This ensures predictable resource allocation and prevents resource conflicts.
"""
    )


if __name__ == '__main__':
    demonstrate_scheduling()
    explain_scheduling()
