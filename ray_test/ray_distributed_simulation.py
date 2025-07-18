#!/usr/bin/env python3

# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""
Ray Distributed Setup Simulation

This example shows how to simulate a distributed Ray cluster on localhost.
We'll create multiple Ray nodes on the same machine to simulate a multi-server setup.
"""

import os
import subprocess
import sys
import time
from typing import Any

import psutil
import ray
import torch

# Configuration
HEAD_PORT = 10001
WORKER_PORT_START = 10002
REDIS_PASSWORD = 'ray_demo_password'


class RayClusterManager:
    """Manages a simulated distributed Ray cluster on localhost."""

    def __init__(self):
        self.head_process = None
        self.worker_processes = []
        self.head_address = None

    def start_head_node(self, num_gpus: int = 2, num_cpus: int = 8) -> str:
        """Start the head node."""
        print('🚀 Starting Ray head node...')

        # Kill any existing Ray processes
        self._cleanup_existing_ray()

        head_cmd = [
            'ray',
            'start',
            '--head',
            f"--port={HEAD_PORT}",
            f"--num-gpus={num_gpus}",
            f"--num-cpus={num_cpus}",
            f"--redis-password={REDIS_PASSWORD}",
            '--include-dashboard=true',
            '--dashboard-port=8265',
        ]

        print(f"Command: {' '.join(head_cmd)}")

        # Start head node
        self.head_process = subprocess.Popen(
            head_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        # Wait a bit for head to start
        time.sleep(3)

        self.head_address = f"ray://127.0.0.1:{HEAD_PORT}"
        print(f"✅ Head node started at {self.head_address}")

        return self.head_address

    def add_worker_node(
        self, node_id: int, num_gpus: int = 0, num_cpus: int = 4
    ) -> bool:
        """Add a worker node to the cluster."""
        print(f"🔧 Adding worker node {node_id}...")

        worker_cmd = [
            'ray',
            'start',
            f"--address={self.head_address}",
            f"--num-gpus={num_gpus}",
            f"--num-cpus={num_cpus}",
            f"--redis-password={REDIS_PASSWORD}",
        ]

        print(f"Command: {' '.join(worker_cmd)}")

        worker_process = subprocess.Popen(
            worker_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        self.worker_processes.append(worker_process)

        # Wait for worker to connect
        time.sleep(2)

        print(f"✅ Worker node {node_id} added")
        return True

    def _cleanup_existing_ray(self):
        """Clean up any existing Ray processes."""
        try:
            subprocess.run(['ray', 'stop', '--force'],
                           capture_output=True,
                           timeout=10)
            time.sleep(1)
        except:
            pass

    def shutdown(self):
        """Shutdown the entire cluster."""
        print('🛑 Shutting down Ray cluster...')

        # Stop all processes
        try:
            subprocess.run(['ray', 'stop', '--force'],
                           capture_output=True,
                           timeout=10)
        except:
            pass

        # Kill processes if still running
        if self.head_process:
            self.head_process.terminate()

        for worker in self.worker_processes:
            worker.terminate()

        print('✅ Cluster shutdown complete')


@ray.remote(num_gpus=1)
class DistributedGPUWorker:
    """A distributed GPU worker that reports its location."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.node_id = ray.get_runtime_context().get_node_id()
        self.gpu_ids = ray.get_gpu_ids()
        self.hostname = os.uname().nodename

    def get_worker_info(self) -> dict[str, Any]:
        """Get information about this worker."""
        return {
            'worker_id':
                self.worker_id,
            'node_id':
                self.node_id,
            'hostname':
                self.hostname,
            'gpu_ids':
                self.gpu_ids,
            'cuda_visible_devices':
                os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
        }

    def distributed_computation(self,
                                matrix_size: int = 1000) -> dict[str, Any]:
        """Perform computation and return node information."""
        start_time = time.time()

        # GPU computation
        device = torch.device('cuda')
        A = torch.randn(matrix_size, matrix_size, device=device)
        B = torch.randn(matrix_size, matrix_size, device=device)
        C = torch.mm(A, B)
        result = torch.trace(C).item()

        execution_time = time.time() - start_time

        return {
            'worker_id': self.worker_id,
            'node_id': self.node_id,
            'hostname': self.hostname,
            'gpu_ids': self.gpu_ids,
            'result': result,
            'execution_time': execution_time,
            'matrix_size': matrix_size,
        }


@ray.remote(num_cpus=1)
def distributed_cpu_task(task_id: int) -> dict[str, Any]:
    """A CPU task that reports which node it's running on."""
    import numpy as np

    start_time = time.time()
    node_id = ray.get_runtime_context().get_node_id()
    hostname = os.uname().nodename

    # CPU computation
    result = np.sum(np.random.randn(500, 500)**2)

    execution_time = time.time() - start_time

    return {
        'task_id': task_id,
        'node_id': node_id,
        'hostname': hostname,
        'result': result,
        'execution_time': execution_time,
        'resource_type': 'CPU',
    }


def demonstrate_cluster_info():
    """Show cluster information and resource distribution."""
    print('\n📊 CLUSTER INFORMATION')
    print('=' * 50)

    # Get cluster resources
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()

    print('Total Cluster Resources:')
    for resource, amount in cluster_resources.items():
        print(f"  {resource}: {amount}")

    print('\nAvailable Resources:')
    for resource, amount in available_resources.items():
        print(f"  {resource}: {amount}")

    # Get node information
    print('\nNodes in Cluster:')
    nodes = ray.nodes()
    for i, node in enumerate(nodes):
        print(f"  Node {i+1}:")
        print(f"    ID: {node['NodeID']}")
        print(f"    Alive: {node['Alive']}")
        print(f"    Resources: {node['Resources']}")


def demonstrate_distributed_gpu_work():
    """Demonstrate distributed GPU work across simulated nodes."""
    print('\n🖥️  DEMO: Distributed GPU Work')
    print('-' * 50)

    # Create GPU workers
    workers = [DistributedGPUWorker.remote(f"gpu_worker_{i}") for i in range(2)]

    # Get worker information
    print('Created GPU workers:')
    worker_info_futures = [
        worker.get_worker_info.remote() for worker in workers
    ]
    worker_infos = ray.get(worker_info_futures)

    for info in worker_infos:
        print(
            f"  {info['worker_id']}: Node {info['node_id'][:8]}, GPU {info['gpu_ids']}"
        )

    # Submit distributed computation
    print('\nSubmitting distributed GPU computations...')
    computation_futures = [
        worker.distributed_computation.remote(matrix_size=1200)
        for worker in workers
    ]

    results = ray.get(computation_futures)

    print('Results:')
    for result in results:
        print(
            f"  {result['worker_id']}: "
            f"Node {result['node_id'][:8]}, "
            f"GPU {result['gpu_ids']}, "
            f"Result: {result['result']:.2f}, "
            f"Time: {result['execution_time']:.2f}s"
        )


def demonstrate_mixed_distributed_work():
    """Demonstrate mixed CPU/GPU work across nodes."""
    print('\n🔄 DEMO: Mixed Distributed Workload')
    print('-' * 50)

    # Submit a mix of CPU and GPU tasks
    cpu_tasks = [distributed_cpu_task.remote(i) for i in range(4)]

    # Create lightweight GPU tasks
    @ray.remote(num_gpus=0.5)
    def light_gpu_task(task_id: int):
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()

        device = torch.device('cuda')
        x = torch.randn(500, 500, device=device)
        result = torch.sum(x * x).item()

        return {
            'task_id': task_id,
            'node_id': node_id,
            'gpu_ids': gpu_ids,
            'result': result,
        }

    gpu_tasks = [light_gpu_task.remote(i + 10) for i in range(3)]

    all_tasks = cpu_tasks + gpu_tasks
    print(
        f"Submitted {len(cpu_tasks)} CPU tasks and {len(gpu_tasks)} GPU tasks"
    )

    start_time = time.time()
    results = ray.get(all_tasks)
    total_time = time.time() - start_time

    print(f"All tasks completed in {total_time:.2f}s")

    # Group results by node
    node_results = {}
    for result in results:
        node_id = result['node_id'][:8]  # Short node ID
        if node_id not in node_results:
            node_results[node_id] = []
        node_results[node_id].append(result)

    print('\nResults by Node:')
    for node_id, node_tasks in node_results.items():
        print(f"  Node {node_id}: {len(node_tasks)} tasks")


def simulate_two_server_setup():
    """Simulate a two-server setup using localhost."""
    print('\n🌐 SIMULATING TWO-SERVER SETUP')
    print('=' * 60)
    print('This simulates Server 1 (head + GPU) and Server 2 (worker + CPU)')

    cluster_manager = RayClusterManager()

    try:
        # Start head node (simulates Server 1 with GPUs)
        head_address = cluster_manager.start_head_node(num_gpus=2, num_cpus=4)

        # Connect Ray client
        print(f"\n🔗 Connecting to distributed cluster at {head_address}")
        ray.init(address=head_address, _redis_password=REDIS_PASSWORD)

        demonstrate_cluster_info()

        # Add worker node (simulates Server 2 with only CPUs)
        cluster_manager.add_worker_node(node_id=1, num_gpus=0, num_cpus=6)

        # Demonstrate distributed functionality
        demonstrate_cluster_info()
        demonstrate_distributed_gpu_work()
        demonstrate_mixed_distributed_work()

        print('\n✨ Distributed simulation completed successfully!')

    except Exception as e:
        print(f"❌ Error in distributed simulation: {e}")

    finally:
        try:
            ray.shutdown()
        except:
            pass
        cluster_manager.shutdown()


def main():
    """Main function to demonstrate distributed Ray setup."""
    print('🎯 Ray Distributed Setup Simulation')
    print('=' * 60)
    print(
        'This example simulates a distributed Ray cluster on a single machine'
    )
    print('to help you understand distributed Ray concepts.')

    # Check if Ray is already running
    try:
        ray.init(address='auto')
        print('⚠️  Ray is already running. Shutting down first...')
        ray.shutdown()
        time.sleep(2)
    except:
        pass

    simulate_two_server_setup()

    print('\n📚 What you learned:')
    print('1. How to start Ray head and worker nodes')
    print('2. How to connect to a distributed Ray cluster')
    print('3. How tasks are distributed across nodes')
    print('4. How to monitor cluster resources and node distribution')
    print('5. How GPU and CPU resources are managed in a distributed setup')

    print('\n🚀 Next steps:')
    print('- Try this on actual multiple servers')
    print('- Experiment with different resource configurations')
    print('- Use Ray Tune for distributed hyperparameter tuning')
    print('- Explore Ray Train for distributed training')


if __name__ == '__main__':
    main()
