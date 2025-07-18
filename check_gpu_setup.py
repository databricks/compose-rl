#!/usr/bin/env python3
"""
GPU Setup Checker

Run this script first to verify your GPU setup is working correctly with Ray.
"""

import ray
import torch
import subprocess
import sys

def check_cuda_available():
    """Check if CUDA is available in PyTorch."""
    print("=== CUDA Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
    else:
        print("❌ CUDA not available! Check your PyTorch installation.")
        return False
    
    return True

def check_nvidia_smi():
    """Check if nvidia-smi is available."""
    print("\n=== nvidia-smi Check ===")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'], 
                              capture_output=True, text=True, check=True)
        print("nvidia-smi output:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ nvidia-smi not found or failed")
        return False

def check_ray_gpu_detection():
    """Check if Ray can detect GPUs."""
    print("\n=== Ray GPU Detection ===")
    
    try:
        ray.init()
        
        resources = ray.cluster_resources()
        print(f"Ray cluster resources: {resources}")
        
        gpu_count = resources.get("GPU", 0)
        if gpu_count >= 2:
            print(f"✅ Ray detected {gpu_count} GPUs")
        elif gpu_count == 1:
            print(f"⚠️  Ray detected only {gpu_count} GPU (expected 2)")
        else:
            print("❌ Ray detected no GPUs")
            
        ray.shutdown()
        return gpu_count >= 2
        
    except Exception as e:
        print(f"❌ Ray initialization failed: {e}")
        return False

def run_simple_gpu_test():
    """Run a simple GPU test to verify everything works."""
    print("\n=== Simple GPU Test ===")
    
    if not torch.cuda.is_available():
        print("❌ Skipping GPU test - CUDA not available")
        return False
    
    try:
        # Test each GPU
        for gpu_id in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{gpu_id}")
            x = torch.randn(100, 100, device=device)
            y = torch.mm(x, x.T)
            print(f"✅ GPU {gpu_id} test passed - tensor shape: {y.shape} on {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all checks."""
    print("Ray + GPU Setup Checker")
    print("=" * 40)
    
    checks = [
        ("CUDA/PyTorch", check_cuda_available),
        ("nvidia-smi", check_nvidia_smi),
        ("Ray GPU Detection", check_ray_gpu_detection),
        ("Simple GPU Test", run_simple_gpu_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to use Ray with GPUs.")
        print("Try running: python ray_gpu_basic.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 