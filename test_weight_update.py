
import asyncio
import os
import subprocess
from composer.utils import dist
from orl_servers.vllm_remote import RemoteVLLMEngine
import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from orl_servers.structs import InferenceEngineConfig, WeightUpdateMeta

from orl_servers.vllm_worker_wrap import stateless_init_process_group


async def _setup_process_groups(
    vllm_engine: RemoteVLLMEngine,
    num_vllm_servers: int,
) -> PyNcclCommunicator:
    """Initialize trainer and vLLM servers' weight-update process group.

    This mirrors the logic used in test_single_controller_vllm.py by:
      - Getting a free TCP port from the master actor
      - Initializing the vLLM servers' NCCL communicators via HTTP
      - Adding a matching process group on the trainer side (rank 0)
    """


    # with socket.socket() as sock:
    #     sock.bind(('', 0))
    #     new_port = sock.getsockname()[1]

    new_port = 9000

    gen_tp_size = 1

    meta = WeightUpdateMeta(
        nccl_master_address="127.0.0.1",
        nccl_master_port=new_port,
        gen_tp_size=gen_tp_size,
        gen_world_size=num_vllm_servers * gen_tp_size,
    )

    # await vllm_engine.ainit_weight_update_group(meta)

    print('Initializing rank 0 process group')
    init_rank_0 = asyncio.to_thread(stateless_init_process_group, 
      "127.0.0.1", new_port, 0, num_vllm_servers * gen_tp_size + 1, torch.cuda.current_device()
    )
    print('init_rank_0', init_rank_0)

    assert init_rank_0 is not None

    # Initialize both sides concurrently
    _, model_update_group = await asyncio.gather(
        vllm_engine.ainit_weight_update_group(meta),
        init_rank_0,
    )
    return model_update_group

def launch_vllm_servers(
    pretrain_model_name: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    num_vllm_servers: int,
    num_train_actors: int,
    max_model_len: int,
    enable_prefix_caching: bool,
) -> tuple[RemoteVLLMEngine, list[subprocess.Popen]]:
    """Launch multiple vLLM HTTP servers and return processes and a RemoteVLLMEngine.

    Servers are started on localhost with ports starting at 8000.
    Each server is assigned a disjoint set of GPUs based on training_world_size.
    """
    processes: list[subprocess.Popen] = []
    addresses: list[str] = []

    for server_idx in range(num_vllm_servers):
        env = os.environ.copy()
        gpu_start = num_train_actors + server_idx * tensor_parallel_size
        gpu_ids = list(range(gpu_start, gpu_start + tensor_parallel_size))
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        port = 8000 + server_idx
        cmd = [
            'orl-vllm-server',
            '--model', pretrain_model_name,
            '--worker-extension-cls', 'orl_servers.vllm_worker_wrap.WorkerWrap',
            '--max-model-len', str(max_model_len),
            '--tensor-parallel-size', str(tensor_parallel_size),
            '--data-parallel-size', str(data_parallel_size),
            '--seed', '1',
            '--enable-prefix-caching' if enable_prefix_caching else '--no-enable-prefix-caching',
            # '--enforce-eager',  # TODO: check if we need to enforce eager
            # '--disable-custom-all-reduce',  # A100 does not like it
            '--port', str(port),
            '--enable-log-requests',
            '--uvicorn-log-level', 'debug',
            '--no-disable-uvicorn-access-log',
        ]
        # Log to stdout and stderr
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        print(' '.join(cmd))
        addresses.append(f'localhost:{port}')

    vllm_engine = RemoteVLLMEngine(
        config=InferenceEngineConfig(
            setup_timeout=180.0,
            request_timeout=300.0,
            request_retries=3,
        ),
        addresses=addresses,
    )
    print(f'Initializing vLLM engine for addresses: {addresses}')
    vllm_engine.initialize()
    print(f'Initialized vLLM engine')
    return vllm_engine, processes


async def main():
    vllm_engine, vllm_procs = None, []
    # try:
    if dist.get_global_rank() == 0:
        vllm_engine, vllm_procs = launch_vllm_servers(
            pretrain_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=1,
            data_parallel_size=1,
            num_vllm_servers=4,
            num_train_actors=4,
            max_model_len=10240,
            enable_prefix_caching=True,
            )
        print('Setting up process groups')
        await _setup_process_groups(vllm_engine, 4)
    # finally:
    #     # Properly shutdown vLLM engine to avoid threading issues
    #     if vllm_engine is not None:
    #         # Give some time for any ongoing operations to complete
    #         await asyncio.sleep(1)
    #     for proc in vllm_procs:
    #         proc.terminate()
    #         proc.wait()  # Wait for process to actually terminate


def run_main():
    """Run the main function with proper asyncio event loop handling."""
    # Set up a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        # Clean up the event loop
        loop.close()

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3 composer -n 4 --world_size 4 test_weight_update.py
    dist.initialize_dist('gpu')
    run_main()

    # print('Initializing process group')
    # stateless_init_process_group("127.0.0.1", 9000, dist.get_global_rank(), 4, torch.cuda.current_device())
    # print('Process group initialized')