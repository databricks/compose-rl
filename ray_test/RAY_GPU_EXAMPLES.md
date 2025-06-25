# Ray GPU Management Examples

This directory contains examples to help you learn Ray for GPU workload management on a single server with 2 GPUs.

## Files Overview

1. **`check_gpu_setup.py`** - Verify your GPU setup
2. **`ray_gpu_basic.py`** - Minimal Ray GPU example
3. **`ray_gpu_patterns.py`** - Advanced GPU management patterns
4. **`test_ray.py`** - Your existing comprehensive example

## Getting Started

### Step 1: Check Your Setup

Before running any examples, verify your GPU setup:

```bash
python check_gpu_setup.py
```

This will check:
- CUDA availability in PyTorch
- nvidia-smi functionality
- Ray GPU detection
- Basic GPU operations

### Step 2: Run the Basic Example

Start with the simplest example:

```bash
python ray_gpu_basic.py
```

This demonstrates:
- Ray initialization
- Basic GPU task creation
- Resource allocation
- Simple parallel execution

### Step 3: Try Advanced Patterns

Explore more sophisticated GPU management:

```bash
python ray_gpu_patterns.py
```

This shows:
- Fractional GPU allocation (0.5 GPU per task)
- Mixed CPU/GPU workloads
- Resource monitoring
- Dynamic scheduling

### Step 4: Study the Complete Example

Your existing `test_ray.py` provides a comprehensive example with:
- Detailed GPU assignment tracking
- Resource visualization
- Error handling
- Best practices

## Key Ray GPU Concepts

### 1. GPU Resource Allocation

```python
# Request 1 full GPU
@ray.remote(num_gpus=1)
def gpu_task():
    pass

# Request 0.5 GPU (allows 2 tasks per GPU)
@ray.remote(num_gpus=0.5)
def light_gpu_task():
    pass
```

### 2. GPU Assignment

Ray automatically:
- Sets `CUDA_VISIBLE_DEVICES` for each task
- Manages GPU memory isolation
- Schedules tasks based on available GPUs

```python
# Inside a Ray task
gpu_ids = ray.get_gpu_ids()  # Get assigned GPU IDs
device = torch.device("cuda")  # PyTorch sees only assigned GPUs
```

### 3. Resource Monitoring

```python
# Check available resources
ray.cluster_resources()    # Total resources
ray.available_resources()  # Currently available
```

## Common Patterns

### Pattern 1: Parallel GPU Tasks
```python
# Launch multiple tasks in parallel
tasks = [gpu_task.remote(i) for i in range(4)]
results = ray.get(tasks)  # Wait for all to complete
```

### Pattern 2: Mixed Workloads
```python
# CPU and GPU tasks running together
cpu_tasks = [cpu_task.remote(i) for i in range(4)]
gpu_tasks = [gpu_task.remote(i) for i in range(2)]
all_results = ray.get(cpu_tasks + gpu_tasks)
```

### Pattern 3: Dynamic Scheduling
```python
# Submit tasks as resources become available
futures = []
for i in range(10):
    future = gpu_task.remote(i)
    futures.append(future)
    if len(futures) >= 2:  # Don't overwhelm the queue
        ray.get(futures[:1])  # Wait for one to complete
        futures = futures[1:]
```

## Monitoring GPU Usage

While running examples, monitor GPU usage:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- GPU utilization changes as tasks start/finish
- Memory allocation per GPU
- Process assignments

## Troubleshooting

### Common Issues

1. **Ray doesn't detect GPUs**
   - Check `nvidia-smi` works
   - Verify CUDA installation
   - Try `ray.init(num_gpus=2)` to force detection

2. **CUDA out of memory**
   - Reduce tensor sizes in examples
   - Use fractional GPU allocation
   - Monitor memory with `nvidia-smi`

3. **Tasks not running in parallel**
   - Check available resources with `ray.available_resources()`
   - Verify you have enough GPUs for your tasks
   - Use `ray.get()` wisely to avoid blocking

### Debug Tips

```python
# Check Ray status
ray.cluster_resources()
ray.available_resources()

# Monitor task execution
import time
start = time.time()
results = ray.get(tasks)
print(f"Execution time: {time.time() - start:.2f}s")
```

## Next Steps

1. **Experiment** with different GPU allocations (0.25, 0.5, 1.0)
2. **Try** mixing CPU and GPU tasks
3. **Monitor** resource usage patterns
4. **Scale up** to more complex workloads
5. **Learn** about Ray Tune for hyperparameter optimization
6. **Explore** Ray Train for distributed training

## Useful Commands

```bash
# Check GPU status
nvidia-smi

# Monitor Ray cluster
ray status

# Ray dashboard (if enabled)
ray dashboard

# Kill Ray processes
ray stop
```

Happy learning with Ray! 🚀 