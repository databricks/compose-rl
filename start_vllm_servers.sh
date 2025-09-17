#!/bin/bash


pkill -9 -f VLLM::EngineCore

CUDA_VISIBLE_DEVICES=4 orl-vllm-server --model Qwen/Qwen2.5-0.5B-Instruct --worker-extension-cls orl_servers.vllm_worker_wrap.WorkerWrap --max-model-len 10240 --tensor-parallel-size 1 --data-parallel-size 1 --seed 1 --enable-prefix-caching --port 8000 &
CUDA_VISIBLE_DEVICES=5 orl-vllm-server --model Qwen/Qwen2.5-0.5B-Instruct --worker-extension-cls orl_servers.vllm_worker_wrap.WorkerWrap --max-model-len 10240 --tensor-parallel-size 1 --data-parallel-size 1 --seed 1 --enable-prefix-caching --port 8001 &
CUDA_VISIBLE_DEVICES=6 orl-vllm-server --model Qwen/Qwen2.5-0.5B-Instruct --worker-extension-cls orl_servers.vllm_worker_wrap.WorkerWrap --max-model-len 10240 --tensor-parallel-size 1 --data-parallel-size 1 --seed 1 --enable-prefix-caching --port 8002 &
CUDA_VISIBLE_DEVICES=7 orl-vllm-server --model Qwen/Qwen2.5-0.5B-Instruct --worker-extension-cls orl_servers.vllm_worker_wrap.WorkerWrap --max-model-len 10240 --tensor-parallel-size 1 --data-parallel-size 1 --seed 1 --enable-prefix-caching --port 8003 &