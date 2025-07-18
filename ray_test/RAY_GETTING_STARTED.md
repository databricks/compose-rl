# Ray GPU Management - Complete Learning Guide

Welcome to Ray GPU management! This guide provides everything you need to learn Ray from scratch, with hands-on examples for both single-server multi-GPU setups and distributed computing simulation.

## 🎯 Learning Path Overview

**You are here:** Complete beginner → Ray GPU expert

```
Step 1: Setup & Verification
    ↓
Step 2: Interactive Learning (Basics)
    ↓
Step 3: Single Server Multi-GPU Patterns
    ↓
Step 4: Distributed Simulation
    ↓
Step 5: Real-World Applications
```

## 📋 Prerequisites

- ✅ Linux system with NVIDIA GPUs
- ✅ CUDA toolkit installed
- ✅ PyTorch with CUDA support
- ✅ Ray installed (`pip install ray[default]`)

## 🚀 Quick Start (3 Commands)

```bash
# 1. Verify your setup
python check_gpu_setup.py

# 2. Learn interactively
python ray_learning_guide.py

# 3. Try advanced patterns
python ray_single_server_multi_gpu.py
```

## 📚 Detailed Learning Steps

### Step 1: Verify Your Setup

First, ensure everything is working:

```bash
python check_gpu_setup.py
```

This checks:
- ✅ CUDA availability in PyTorch
- ✅ nvidia-smi functionality
- ✅ Ray GPU detection
- ✅ Basic GPU operations

**Expected output:** All checks should pass with "🎉 All checks passed!"

### Step 2: Interactive Learning (START HERE!)

Perfect for complete beginners:

```bash
python ray_learning_guide.py
```

**What you'll learn:**
- Ray basic concepts (remote functions, actors)
- GPU resource allocation (full vs fractional)
- Tasks vs Actors differences
- Resource monitoring

**Duration:** 10-15 minutes (interactive)

### Step 3: Single Server Multi-GPU Patterns

Advanced patterns on your single server with 2 GPUs:

```bash
python ray_single_server_multi_gpu.py
```

**What you'll see:**
- 🚀 GPU Actors (long-lived workers)
- 🔄 Fractional GPU allocation (0.5 GPU per task)
- 🔀 Mixed CPU/GPU workloads
- ⚡ Dynamic scheduling patterns

**Duration:** 5-10 minutes (automated demos)

### Step 4: Distributed Simulation

Simulate multi-server setup on localhost:

```bash
python ray_distributed_simulation.py
```

**What you'll learn:**
- Starting Ray head and worker nodes
- Connecting to distributed clusters
- Task distribution across nodes
- Node-level resource management

**Note:** This simulates "Server 1" (head + GPUs) and "Server 2" (worker + CPUs)

## 🔍 Monitoring Your Ray Cluster

### During Learning

While running examples, monitor GPU usage:

```bash
# Terminal 1: Run your Ray script
python ray_learning_guide.py

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi
```

### Ray Dashboard

When Ray is running, visit the dashboard:
```
http://localhost:8265
```

Shows:
- 📊 Resource utilization
- 🎯 Task execution timeline
- 🖥️  Node status and health
- 📈 Performance metrics

## 📖 Core Concepts Reference

### GPU Resource Allocation

```python
# Full GPU (exclusive access)
@ray.remote(num_gpus=1)
def gpu_task():
    pass

# Fractional GPU (shared access)
@ray.remote(num_gpus=0.5)  # 2 tasks per GPU
def light_gpu_task():
    pass
```

### Tasks vs Actors

```python
# TASK: Stateless function
@ray.remote(num_gpus=1)
def process_data(data):
    return result

# ACTOR: Stateful class
@ray.remote(num_gpus=1)
class DataProcessor:
    def __init__(self):
        self.model = load_model()

    def process(self, data):
        return self.model(data)
```

### Key Ray Functions

```python
# Submit work
future = task.remote(data)
actor = Actor.remote()

# Get results
result = ray.get(future)
results = ray.get([future1, future2])

# Monitor resources
ray.cluster_resources()    # Total
ray.available_resources()  # Available now
```

## 🛠️ Common Patterns

### Pattern 1: Parallel GPU Processing
```python
@ray.remote(num_gpus=1)
def train_model(config):
    # Your GPU training code
    pass

# Train multiple models in parallel
configs = [config1, config2]
futures = [train_model.remote(c) for c in configs]
results = ray.get(futures)
```

### Pattern 2: Mixed Workloads
```python
# Mix CPU preprocessing with GPU training
cpu_tasks = [preprocess.remote(data) for data in dataset]
processed_data = ray.get(cpu_tasks)

gpu_tasks = [train.remote(data) for data in processed_data]
models = ray.get(gpu_tasks)
```

### Pattern 3: Pipeline Processing
```python
@ray.remote(num_cpus=1)
def preprocess(data):
    return cleaned_data

@ray.remote(num_gpus=0.5)
def inference(data):
    return predictions

# Pipeline: preprocess → inference
for data_batch in dataset:
    clean_data = preprocess.remote(data_batch)
    predictions = inference.remote(clean_data)
    results.append(predictions)
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**Issue:** Ray doesn't detect GPUs
```bash
# Solution: Force GPU detection
ray.init(num_gpus=2)

# Or check GPU visibility
nvidia-smi
```

**Issue:** CUDA out of memory
```bash
# Solution: Use fractional GPUs
@ray.remote(num_gpus=0.5)  # Instead of 1.0

# Or reduce tensor sizes
x = torch.randn(1000, 1000)  # Instead of 5000x5000
```

**Issue:** Tasks not running in parallel
```python
# Solution: Check available resources
print(ray.available_resources())

# Don't block waiting for results too early
futures = [task.remote(i) for i in range(10)]
# Do other work here...
results = ray.get(futures)  # Wait at the end
```

**Issue:** Ray processes hanging
```bash
# Solution: Clean shutdown
ray stop --force
pkill -f ray
```

## 🎯 What's Next?

After completing this guide, explore:

### Advanced Ray Features
- **Ray Tune:** Hyperparameter optimization
- **Ray Train:** Distributed training
- **Ray Serve:** Model serving
- **Ray Data:** Large-scale data processing

### Real Distributed Setup
Once comfortable with localhost simulation:

```bash
# Server 1 (head node)
ray start --head --port=10001 --num-gpus=2

# Server 2 (worker node)
ray start --address=192.168.1.100:10001 --num-gpus=1
```

### Production Considerations
- Resource management policies
- Fault tolerance and recovery
- Monitoring and logging
- Auto-scaling strategies

## 📁 Files in This Learning Package

| File | Purpose | When to Use |
|------|---------|-------------|
| `check_gpu_setup.py` | Verify system setup | **Start here** - before anything else |
| `ray_learning_guide.py` | Interactive beginner tutorial | **Step 2** - core concepts |
| `ray_single_server_multi_gpu.py` | Advanced single-server patterns | **Step 3** - practical patterns |
| `ray_distributed_simulation.py` | Localhost distributed simulation | **Step 4** - distributed concepts |
| `ray_gpu_basic.py` | Simple working example | Reference/quick test |
| `RAY_GPU_EXAMPLES.md` | Original documentation | Additional reference |

## 🎉 Success Metrics

You'll know you've mastered Ray GPU management when you can:

✅ Set up Ray clusters (single and distributed)
✅ Choose between tasks and actors appropriately
✅ Allocate GPU resources efficiently (full vs fractional)
✅ Monitor and debug resource usage
✅ Design efficient parallel workflows
✅ Handle mixed CPU/GPU workloads

## 🆘 Getting Help

- 📖 [Official Ray Documentation](https://docs.ray.io/)
- 💬 [Ray Discourse Forum](https://discuss.ray.io/)
- 🐛 [Ray GitHub Issues](https://github.com/ray-project/ray/issues)
- 📺 [Ray YouTube Tutorials](https://www.youtube.com/c/RayProjectIO)

---

**Happy learning!** 🚀 Start with `python ray_learning_guide.py` and work your way through the examples.
