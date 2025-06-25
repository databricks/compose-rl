#!/usr/bin/env python3
"""
Ray GPU Management - Advanced Patterns

This example demonstrates different GPU management patterns in Ray:
1. Fractional GPU allocation
2. Dynamic task scheduling
3. Resource monitoring
4. Error handling
"""

import ray
import torch
import time
import psutil

# Pattern 1: Fractional GPU usage (0.5 GPU per task)
@ray.remote(num_gpus=0.5)
def light_gpu_task(task_id: int):
    """Task that only needs half a GPU - allows 4 tasks on 2 GPUs."""
    gpu_ids = ray.get_gpu_ids()
    device = torch.device("cuda")
    
    print(f"Light task {task_id}: Using GPU fraction on {gpu_ids}")
    
    # Lighter computation
    x = torch.randn(500, 500, device=device)
    x = torch.mm(x, x.T)
    time.sleep(1)
    
    return f"Light task {task_id} done"

# Pattern 2: Full GPU usage
@ray.remote(num_gpus=1)
def heavy_gpu_task(task_id: int):
    """Task that needs a full GPU."""
    gpu_ids = ray.get_gpu_ids()
    device = torch.device("cuda")
    
    print(f"Heavy task {task_id}: Using full GPU {gpu_ids[0]}")
    
    # Heavier computation
    x = torch.randn(2000, 2000, device=device)
    for _ in range(5):
        x = torch.mm(x, x.T)
    time.sleep(2)
    
    return f"Heavy task {task_id} done on GPU {gpu_ids[0]}"

# Pattern 3: CPU task for comparison
@ray.remote
def cpu_task(task_id: int):
    """Task that runs on CPU only."""
    print(f"CPU task {task_id}: Running on CPU")
    
    # CPU computation
    x = torch.randn(1000, 1000)
    x = torch.mm(x, x.T)
    time.sleep(1)
    
    return f"CPU task {task_id} done"

# Pattern 4: Resource monitoring task
@ray.remote
def monitor_resources():
    """Monitor system resources while tasks are running."""
    resources = ray.cluster_resources()
    available = ray.available_resources()
    
    return {
        "total_gpus": resources.get("GPU", 0),
        "available_gpus": available.get("GPU", 0),
        "total_cpus": resources.get("CPU", 0),
        "available_cpus": available.get("CPU", 0),
        "memory_usage": psutil.virtual_memory().percent
    }

def demonstrate_gpu_patterns():
    """Demonstrate different GPU allocation patterns."""
    
    print("=== Ray GPU Patterns Demo ===\n")
    
    # Initialize Ray
    ray.init()
    
    # Check available resources
    print("Initial resources:", ray.cluster_resources())
    print("Available resources:", ray.available_resources())
    print()
    
    # Pattern 1: Run multiple light tasks (fractional GPU)
    print("1. Running 4 light tasks (0.5 GPU each) - should run 4 concurrent on 2 GPUs")
    light_tasks = [light_gpu_task.remote(i) for i in range(4)]
    
    # Pattern 2: Run heavy tasks (full GPU)
    print("2. Running 2 heavy tasks (1 GPU each)")
    heavy_tasks = [heavy_gpu_task.remote(i) for i in range(2)]
    
    # Pattern 3: Run CPU tasks alongside
    print("3. Running CPU tasks in parallel")
    cpu_tasks = [cpu_task.remote(i) for i in range(3)]
    
    # Pattern 4: Monitor resources while tasks run
    monitor_task = monitor_resources.remote()
    
    # Wait for light tasks
    print("\nWaiting for light tasks...")
    light_results = ray.get(light_tasks)
    print("Light tasks results:", light_results)
    
    # Check resources mid-execution
    mid_resources = ray.get(monitor_task)
    print("Mid-execution resources:", mid_resources)
    
    # Wait for remaining tasks
    print("\nWaiting for heavy and CPU tasks...")
    heavy_results = ray.get(heavy_tasks)
    cpu_results = ray.get(cpu_tasks)
    
    print("Heavy tasks results:", heavy_results)
    print("CPU tasks results:", cpu_results)
    
    # Final resource check
    final_monitor = monitor_resources.remote()
    final_resources = ray.get(final_monitor)
    print("Final resources:", final_resources)
    
    ray.shutdown()

if __name__ == "__main__":
    demonstrate_gpu_patterns() 