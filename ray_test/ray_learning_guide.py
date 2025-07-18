#!/usr/bin/env python3

# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""
Ray GPU Learning Guide - Getting Started Script

This script helps beginners understand Ray GPU management concepts
through interactive examples and clear explanations.
"""

import time

import ray
import torch


def step_1_basic_concepts():
    """Step 1: Understanding Ray basic concepts."""
    print('\n' + '=' * 60)
    print('🎓 STEP 1: RAY BASIC CONCEPTS')
    print('=' * 60)

    print(
        """
Ray is a distributed computing framework that helps you:
1. Parallelize your Python code across multiple cores/machines
2. Manage GPU resources automatically
3. Scale from single machine to clusters seamlessly

Key concepts:
- @ray.remote: Decorator to make functions/classes distributed
- ray.get(): Wait for and retrieve results from remote tasks
- ray.put(): Store large objects in shared memory
- Actors: Stateful workers that persist across tasks
"""
    )

    # Simple example
    @ray.remote
    def simple_task(x):
        return x * x

    print('Example: Simple remote function')
    print('@ray.remote')
    print('def simple_task(x):')
    print('    return x * x')

    # Execute
    future = simple_task.remote(5)
    result = ray.get(future)
    print(f"\nResult: simple_task.remote(5) = {result}")


def step_2_gpu_resource_management():
    """Step 2: Understanding GPU resource management."""
    print('\n' + '=' * 60)
    print('🎮 STEP 2: GPU RESOURCE MANAGEMENT')
    print('=' * 60)

    print(
        """
Ray automatically manages GPU allocation:

1. Full GPU allocation: @ray.remote(num_gpus=1)
   - Task gets exclusive access to 1 GPU
   - Ray sets CUDA_VISIBLE_DEVICES automatically

2. Fractional GPU allocation: @ray.remote(num_gpus=0.5)
   - Multiple tasks can share the same GPU
   - Useful for lightweight GPU work

3. Ray handles scheduling based on available resources
"""
    )

    @ray.remote(num_gpus=1)
    def gpu_task(task_id):
        gpu_ids = ray.get_gpu_ids()
        device = torch.device('cuda')
        x = torch.randn(100, 100, device=device)
        return {'task_id': task_id, 'gpu_ids': gpu_ids, 'shape': list(x.shape)}

    print('Example: GPU task')
    print('@ray.remote(num_gpus=1)')
    print('def gpu_task(task_id):')
    print('    gpu_ids = ray.get_gpu_ids()')
    print("    device = torch.device('cuda')")
    print('    x = torch.randn(100, 100, device=device)')
    print("    return {'task_id': task_id, 'gpu_ids': gpu_ids}")

    # Execute on both GPUs
    tasks = [gpu_task.remote(i) for i in range(2)]
    results = ray.get(tasks)

    print(f"\nResults from 2 GPU tasks:")
    for result in results:
        print(
            f"  Task {result['task_id']}: GPU {result['gpu_ids']}, Tensor {result['shape']}"
        )


def step_3_actors_vs_tasks():
    """Step 3: Understanding the difference between actors and tasks."""
    print('\n' + '=' * 60)
    print('🎭 STEP 3: ACTORS VS TASKS')
    print('=' * 60)

    print(
        """
Tasks vs Actors:

TASKS (@ray.remote functions):
- Stateless and lightweight
- Good for simple computations
- GPU allocated only during execution
- No memory between calls

ACTORS (@ray.remote classes):
- Stateful workers with persistent memory
- Good for complex workflows
- GPU held for the lifetime of the actor
- Can maintain state between method calls
"""
    )

    @ray.remote(num_gpus=0.5)
    class GPUActor:

        def __init__(self):
            self.gpu_ids = ray.get_gpu_ids()
            self.device = torch.device('cuda')
            self.counter = 0

        def process(self, data_size=500):
            self.counter += 1
            x = torch.randn(data_size, data_size, device=self.device)
            y = torch.mm(x, x.T)
            return {
                'call_count': self.counter,
                'gpu_ids': self.gpu_ids,
                'result': torch.trace(y).item(),
            }

    print('Example: GPU Actor')
    print('@ray.remote(num_gpus=0.5)')
    print('class GPUActor:')
    print('    def __init__(self):')
    print('        self.gpu_ids = ray.get_gpu_ids()')
    print("        self.device = torch.device('cuda')")
    print('        self.counter = 0')

    # Create actors (4 actors, 2 per GPU with 0.5 GPU each)
    actors = [GPUActor.remote() for _ in range(4)]

    # Call methods multiple times
    futures = []
    for actor in actors:
        for _ in range(2):  # 2 calls per actor
            futures.append(actor.process.remote())

    results = ray.get(futures)

    print(f"\nResults from {len(actors)} actors, each called twice:")
    for i, result in enumerate(results):
        print(
            f"  Call {i+1}: GPU {result['gpu_ids']}, Count: {result['call_count']}, Result: {result['result']:.2f}"
        )


def step_4_monitoring_resources():
    """Step 4: Understanding resource monitoring."""
    print('\n' + '=' * 60)
    print('📊 STEP 4: MONITORING RESOURCES')
    print('=' * 60)

    print(
        """
Ray provides several ways to monitor resources:

1. ray.cluster_resources() - Total resources in cluster
2. ray.available_resources() - Currently available resources
3. ray.nodes() - Information about cluster nodes
4. Ray Dashboard - Web UI for monitoring (http://localhost:8265)
"""
    )

    print('Current cluster state:')
    print(f"  Total resources: {ray.cluster_resources()}")
    print(f"  Available resources: {ray.available_resources()}")

    # Show how resources change during execution
    @ray.remote(num_gpus=1)
    def blocking_gpu_task():
        print(f"  📍 Task started on GPU {ray.get_gpu_ids()}")
        time.sleep(3)  # Hold GPU for 3 seconds
        return 'done'

    print('\nWatching resources during task execution...')
    print('Available before task:', ray.available_resources().get('GPU', 0))

    future = blocking_gpu_task.remote()
    time.sleep(0.5)  # Give task time to start
    print('Available during task:', ray.available_resources().get('GPU', 0))

    ray.get(future)
    print('Available after task: ', ray.available_resources().get('GPU', 0))


def interactive_learning_session():
    """Run an interactive learning session."""
    print('🎯 RAY GPU MANAGEMENT - INTERACTIVE LEARNING')
    print('=' * 70)

    print(
        """
Welcome to Ray GPU Management Learning!

This script will teach you Ray concepts step by step.
Each step builds on the previous one.

You have 2 NVIDIA A100 GPUs available for learning.
"""
    )

    # Initialize Ray
    print('🚀 Initializing Ray...')
    ray.init(num_gpus=2)
    print(f"✅ Ray initialized with resources: {ray.cluster_resources()}")

    try:
        step_1_basic_concepts()

        input('\nPress Enter to continue to Step 2...')
        step_2_gpu_resource_management()

        input('\nPress Enter to continue to Step 3...')
        step_3_actors_vs_tasks()

        input('\nPress Enter to continue to Step 4...')
        step_4_monitoring_resources()

        print('\n' + '=' * 70)
        print('🎉 CONGRATULATIONS!')
        print('=' * 70)
        print(
            """
You've learned the fundamentals of Ray GPU management:

✅ Basic Ray concepts (remote functions, ray.get)
✅ GPU resource allocation (full and fractional)
✅ Difference between tasks and actors
✅ Resource monitoring and observation

Next steps to continue learning:
1. Run 'python ray_single_server_multi_gpu.py' for advanced patterns
2. Run 'python ray_distributed_simulation.py' for distributed concepts
3. Try the Ray dashboard at http://localhost:8265
4. Explore Ray Tune for hyperparameter optimization
5. Look into Ray Train for distributed training

Happy learning! 🚀
"""
        )

    except KeyboardInterrupt:
        print('\n\n👋 Learning session interrupted. Thanks for trying Ray!')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    interactive_learning_session()
