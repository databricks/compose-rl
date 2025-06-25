import ray
import torch
import time
import socket # Import socket to see which node we're on

# The task definition is IDENTICAL to the previous example.
# No changes are needed here.
@ray.remote(num_gpus=1)
def use_gpu_task(task_id: int):
    gpu_ids = ray.get_gpu_ids()
    physical_gpu_id = gpu_ids[0]
    
    # Let's also get the hostname of the node Ray scheduled us on.
    # In this simulation, it will be the same hostname, but Ray
    # internally treats them as distinct nodes.
    node_id = ray.get_runtime_context().get_node_id()
    
    print(
        f"-> Task {task_id} starting."
        f" Ray assigned me physical GPU: {physical_gpu_id}"
        f" on Node ID: {node_id}"
    )

    device = torch.device("cuda")
    tensor = torch.randn(2000, 2000, device=device)

    for i in range(5):
        tensor = tensor @ tensor
        time.sleep(0.5)

    print(f"<- Task {task_id} finished on GPU {physical_gpu_id}.")
    return f"Task {task_id} ran on GPU {physical_gpu_id} on Node {node_id}"


# =================================================================
# MAIN SCRIPT
# =================================================================
if __name__ == "__main__":
    # CRITICAL CHANGE: Connect to the existing Ray cluster.
    # 'auto' tells Ray to find the running cluster from environment variables
    # that `ray start` sets up.
    ray.init()

    print("Python script connected to Ray Cluster.")
    print("Cluster resources:", ray.cluster_resources())

    # The rest of the logic is the same.
    print("\nLaunching 4 GPU tasks on our 2-node, 2-GPU cluster...")
    task_refs = []
    for i in range(4):
        ref = use_gpu_task.remote(i)
        task_refs.append(ref)

    results = ray.get(task_refs)

    print("\n--- All tasks completed! ---")
    print("Results:", results)

    # We don't call ray.shutdown() here, because we want to leave the
    # cluster running. We will stop it manually from the terminal.
    print("\nScript finished. The Ray cluster is still running.")