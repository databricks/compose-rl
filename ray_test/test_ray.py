import ray
import torch
import time


# =================================================================
# 1. THE RAY TASK: A function that will run on a single GPU
# =================================================================
# The decorator is the key: it tells Ray this task requires 1 GPU.
@ray.remote(num_gpus=1)
def use_gpu_task(task_id: int):
    """
    A simple Ray task that simulates work on a GPU.
    """
    # Ray automatically sets the CUDA_VISIBLE_DEVICES environment variable
    # for this worker process, so torch.cuda.current_device() will
    # correspond to the GPU Ray assigned.
    
    # Let's get the physical GPU ID that Ray assigned to this task.
    gpu_ids = ray.get_gpu_ids()
    physical_gpu_id = gpu_ids[0]

    print(f"-> Task {task_id} starting. Ray assigned me physical GPU: {physical_gpu_id}")

    # Create a tensor and move it to the assigned GPU.
    # PyTorch will only see the single GPU that Ray allocated.
    device = torch.device("cuda")
    tensor = torch.randn(2000, 2000, device=device)

    # Perform some work to make the GPU busy.
    for i in range(5):
        tensor = tensor @ tensor
        time.sleep(0.5) # Sleep to make it easier to see in nvidia-smi
        print(f"   Task {task_id}, iteration {i+1}, on device: {tensor.device}")

    print(f"<- Task {task_id} finished on GPU {physical_gpu_id}.")
    
    # Return the ID of the GPU we used.
    return f"Task {task_id} ran on GPU {physical_gpu_id}"


# =================================================================
# 2. MAIN SCRIPT: Initialize Ray and launch the tasks
# =================================================================
if __name__ == "__main__":
    # Start Ray. Ray will automatically detect the 2 GPUs.
    # You could also be explicit: ray.init(num_gpus=2)
    ray.init()

    print("Ray Initialized.")
    print("Cluster resources:", ray.cluster_resources())

    # Verify that Ray sees our GPUs
    if ray.cluster_resources().get("GPU", 0) < 2:
        print("!!! WARNING: Ray did not detect 2 GPUs. Exiting.")
        ray.shutdown()
        exit()

    # We have 2 GPUs, so let's launch 4 tasks.
    # Ray will run 2 tasks concurrently, and queue the other 2
    # until the first ones finish.
    print("\nLaunching 4 GPU tasks on 2 available GPUs...")
    task_refs = []
    for i in range(4):
        # .remote() immediately returns a future (a reference to the result)
        # and executes the task in the background.
        ref = use_gpu_task.remote(i)
        task_refs.append(ref)

    # Block until all tasks are complete and get the results.
    results = ray.get(task_refs)

    print("\n--- All tasks completed! ---")
    print("Results:", results)

    # Shut down Ray
    ray.shutdown()