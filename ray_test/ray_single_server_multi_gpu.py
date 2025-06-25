#!/usr/bin/env python3
"""
Ray Single Server Multi-GPU Example

This example demonstrates Ray GPU management on a single server with multiple GPUs.
Shows various patterns: full GPU allocation, fractional allocation, and mixed workloads.
"""

import ray
import torch
import time
import numpy as np
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class GPUWorker:
    """A Ray actor that holds a full GPU for the duration of its lifetime."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.gpu_ids = ray.get_gpu_ids()
        self.device = torch.device("cuda")
        logger.info(f"Worker {worker_id} initialized on GPU {self.gpu_ids}")
    
    def matrix_multiply(self, size: int = 2000, iterations: int = 5) -> Dict[str, Any]:
        """Perform matrix multiplication to simulate GPU work."""
        start_time = time.time()
        
        # Create random matrices on GPU
        A = torch.randn(size, size, device=self.device)
        B = torch.randn(size, size, device=self.device)
        
        results = []
        for i in range(iterations):
            C = torch.mm(A, B)
            results.append(torch.trace(C).item())
            
        end_time = time.time()
        
        return {
            "worker_id": self.worker_id,
            "gpu_ids": self.gpu_ids,
            "execution_time": end_time - start_time,
            "results": results[:3],  # Just first 3 for brevity
            "tensor_shape": list(C.shape)
        }
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            gpu_id = self.gpu_ids[0]
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
            return {
                "gpu_id": gpu_id,
                "allocated_gb": allocated,
                "cached_gb": cached
            }
        return {}

@ray.remote(num_gpus=0.5)
def lightweight_gpu_task(task_id: int, work_size: int = 1000) -> Dict[str, Any]:
    """A task that uses half a GPU - allows 2 tasks per GPU."""
    start_time = time.time()
    gpu_ids = ray.get_gpu_ids()
    
    device = torch.device("cuda")
    x = torch.randn(work_size, work_size, device=device)
    
    # Simulate some computation
    for _ in range(3):
        x = torch.relu(x @ x.T)
    
    end_time = time.time()
    
    return {
        "task_id": task_id,
        "gpu_ids": gpu_ids,
        "execution_time": end_time - start_time,
        "final_mean": x.mean().item()
    }

@ray.remote(num_cpus=1)
def cpu_task(task_id: int) -> Dict[str, Any]:
    """A CPU-only task to demonstrate mixed workloads."""
    start_time = time.time()
    
    # CPU computation
    result = np.sum(np.random.randn(1000, 1000) ** 2)
    time.sleep(1)  # Simulate work
    
    end_time = time.time()
    
    return {
        "task_id": task_id,
        "execution_time": end_time - start_time,
        "result": result,
        "resource_type": "CPU"
    }

def print_resources():
    """Print current Ray cluster resources."""
    print("\n" + "="*50)
    print("RAY CLUSTER RESOURCES")
    print("="*50)
    print(f"Total resources: {ray.cluster_resources()}")
    print(f"Available resources: {ray.available_resources()}")
    print("="*50)

def demo_gpu_actors():
    """Demonstrate GPU actors (long-lived GPU workers)."""
    print("\n🚀 DEMO 1: GPU Actors (Long-lived Workers)")
    print("-" * 50)
    
    # Create 2 GPU workers (one per GPU)
    workers = [GPUWorker.remote(i) for i in range(2)]
    
    # Submit work to both workers
    futures = []
    for i, worker in enumerate(workers):
        future = worker.matrix_multiply.remote(size=1500, iterations=3)
        futures.append(future)
    
    print("Submitted work to GPU actors...")
    results = ray.get(futures)
    
    for result in results:
        print(f"  Worker {result['worker_id']}: GPU {result['gpu_ids']}, "
              f"Time: {result['execution_time']:.2f}s")
    
    # Check memory usage
    memory_futures = [worker.get_gpu_memory_usage.remote() for worker in workers]
    memory_results = ray.get(memory_futures)
    
    for mem in memory_results:
        print(f"  GPU {mem['gpu_id']}: {mem['allocated_gb']:.2f}GB allocated, "
              f"{mem['cached_gb']:.2f}GB cached")
    
    return workers

def demo_fractional_gpu():
    """Demonstrate fractional GPU allocation."""
    print("\n🔄 DEMO 2: Fractional GPU Tasks (0.5 GPU each)")
    print("-" * 50)
    
    # Launch 4 tasks with 0.5 GPU each (2 per GPU)
    tasks = [lightweight_gpu_task.remote(i, work_size=800) for i in range(4)]
    
    print("Submitted 4 tasks with 0.5 GPU each...")
    results = ray.get(tasks)
    
    for result in results:
        print(f"  Task {result['task_id']}: GPU {result['gpu_ids']}, "
              f"Time: {result['execution_time']:.2f}s")

def demo_mixed_workload():
    """Demonstrate mixed CPU and GPU workloads."""
    print("\n🔀 DEMO 3: Mixed CPU and GPU Workloads")
    print("-" * 50)
    
    # Mix of CPU and GPU tasks
    cpu_tasks = [cpu_task.remote(i) for i in range(3)]
    gpu_tasks = [lightweight_gpu_task.remote(i+10, work_size=600) for i in range(3)]
    
    all_tasks = cpu_tasks + gpu_tasks
    print(f"Submitted {len(cpu_tasks)} CPU tasks and {len(gpu_tasks)} GPU tasks...")
    
    start_time = time.time()
    results = ray.get(all_tasks)
    total_time = time.time() - start_time
    
    print(f"All tasks completed in {total_time:.2f}s")
    
    # Separate results
    cpu_results = [r for r in results if r.get('resource_type') == 'CPU']
    gpu_results = [r for r in results if 'gpu_ids' in r]
    
    print(f"  CPU tasks: {len(cpu_results)} completed")
    print(f"  GPU tasks: {len(gpu_results)} completed")

def demo_dynamic_scheduling():
    """Demonstrate dynamic task scheduling based on resource availability."""
    print("\n⚡ DEMO 4: Dynamic Scheduling")
    print("-" * 50)
    
    # Submit tasks gradually and monitor resource usage
    completed_tasks = []
    pending_tasks = []
    
    for i in range(8):
        task = lightweight_gpu_task.remote(i, work_size=500)
        pending_tasks.append(task)
        
        # Check if we should wait for some tasks to complete
        if len(pending_tasks) >= 4:  # Don't overwhelm the queue
            # Wait for at least one task to complete
            ready, pending_tasks = ray.wait(pending_tasks, num_returns=1)
            completed_tasks.extend(ray.get(ready))
            print(f"  Completed {len(completed_tasks)} tasks, "
                  f"{len(pending_tasks)} still pending")
    
    # Wait for remaining tasks
    if pending_tasks:
        completed_tasks.extend(ray.get(pending_tasks))
    
    print(f"Dynamic scheduling completed: {len(completed_tasks)} total tasks")

def main():
    """Main function demonstrating various Ray GPU patterns."""
    print("🎯 Ray Single Server Multi-GPU Demo")
    print("=" * 60)
    
    # Initialize Ray
    ray.init(num_gpus=2)  # Explicitly specify 2 GPUs
    
    print_resources()
    
    # Run all demos
    workers = demo_gpu_actors()
    print_resources()
    
    demo_fractional_gpu()
    print_resources()
    
    demo_mixed_workload()
    print_resources()
    
    demo_dynamic_scheduling()
    print_resources()
    
    print("\n🎉 All demos completed!")
    print("\nKey Takeaways:")
    print("1. GPU Actors: Long-lived workers for persistent GPU allocation")
    print("2. Fractional GPUs: Share GPUs between multiple light tasks")
    print("3. Mixed Workloads: Combine CPU and GPU tasks efficiently")
    print("4. Dynamic Scheduling: Adapt to resource availability")
    
    # Cleanup
    ray.shutdown()

if __name__ == "__main__":
    main() 