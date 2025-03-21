# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import subprocess
import sys
import time

import ray
import torch
import torch.distributed as dist
from kubernetes import client, config
from llmfoundry.command_utils import train_from_yaml
from omegaconf import OmegaConf as om

from vllm_utils import create_vllm_engines, init_process_group

# Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Handles both color codes and formatting like bold, underline, etc.
    """
    ansi_pattern = r'\x1b\[[0-9;]*[a-zA-Z]'
    return re.sub(ansi_pattern, '', text)


def parse_ray_local_ip(log_output: str) -> str:
    """Parse the local node IP from Ray runtime startup logs.

    Args:
        log_output (str): The complete log output from Ray startup

    Returns:
        str: The extracted local node IP address

    Raises:
        ValueError: If no local node IP is found in the logs
    """
    # First clean the ANSI escape sequences
    cleaned_output = strip_ansi(log_output)

    # Then look for the IP
    ip_pattern = r'Local node IP: (\d+\.\d+\.\d+\.\d+)'
    match = re.search(ip_pattern, cleaned_output)

    if match:
        return match.group(1)
    else:
        raise ValueError('No local node IP found in the logs')


def broadcast_string(message: str, src_rank: int):
    encoded = message.encode('utf-8')
    length_tensor = torch.LongTensor([len(encoded)])
    dist.broadcast(length_tensor, src=src_rank)

    data_tensor = torch.ByteTensor(list(encoded)) if dist.get_rank() == src_rank else \
                  torch.ByteTensor([0] * length_tensor.item())
    dist.broadcast(data_tensor, src=src_rank)

    return data_tensor.cpu().numpy().tobytes().decode('utf-8')


def recv_string(src: int) -> str:
    """Receive the length of a string, then receive the actual bytes and decode
    them."""
    # 1) Receive length
    length_tensor = torch.LongTensor([0])
    dist.recv(length_tensor, src=src)

    # 2) Receive the bytes
    data_tensor = torch.ByteTensor(length_tensor.item())
    dist.recv(data_tensor, src=src)

    # 3) Decode
    return data_tensor.numpy().tobytes().decode('utf-8')


def broadcast_to_vllm(model, vllm_engines):
    # avoid OOM
    torch.cuda.empty_cache()
    count, num_params = 0, len(list(model.named_parameters()))
    refss = []
    for name, param in model.named_parameters():
        count += 1
        shape = param.shape
        refs = [
            engine.update_weight.remote(
                name,
                dtype=param.dtype,
                shape=shape,
                empty_cache=count == num_params,
            ) for engine in vllm_engines
        ]
        refss.extend(refs)
        torch.distributed.broadcast(
            param.data,
            0,
            group=model_update_group,
        )
    ray.get(refss)

def start_ray_nodes():
    rank = int(os.getenv('NODE_RANK'))
    world_size = int(os.getenv('NUM_NODES'))
    print("starting ray nodes on master port: ", os.getenv('MASTER_PORT'))

    train_num_nodes = os.getenv('TRAIN_NUM_NODES', None)

    if train_num_nodes is not None and rank != 0:
        log.info(
            "On a training node that isn't the master node no need to start ray.",
        )
        return

    vars_to_check = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'NODE_RANK']
    for var in vars_to_check:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    node_rank = os.getenv('NODE_RANK', None)
    if node_rank is None:
        raise ValueError('NODE_RANK must be set')
    node_rank = int(node_rank)

    log.info('Finished waiting.')

    if node_rank == 0:
        result = subprocess.run(
            ['ray', 'start', '--head', '--port=6379'],
            check=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            log.debug('Error starting Ray!')
            log.debug(f'STDOUT: {result.stdout}')
            log.debug(f'STDERR: {result.stderr}')
        log.info(repr(result.stdout))

        ip = parse_ray_local_ip(result.stdout)
        log.info(f'On rank 0 IP is: {ip}')

        # Send the local node IP to other ranks
        broadcast_string(ip, src_rank=0)

        ray.init()
        # Wait for all ray clusters to start
        dist.barrier()

        log.info('Waiting 10 seconds for all ray clusters to start.')

        log.info('On rank 0 printing all possible nodes')
        for node in ray.nodes():
            log.info(f"Node: {node['NodeManagerAddress']}")
            log.info(f"Resources: {node['Resources']}")
            log.info(f"Alive: {node['Alive']}\n")
            # print(f"Node: {node['NodeManagerAddress']}")
            # print(f"Resources: {node['Resources']}")
            # print(f"Alive: {node['Alive']}\n")

    elif node_rank > 0:

        # Ranks 1..(world_size-1) -> receive message from rank 0
        print('In inference node trying to receive string')
        incoming_msg = broadcast_string('', src_rank=0)
        log.info(f'[Rank {rank}] Received message from rank 0: {incoming_msg}')
        print('incoming message is: ', incoming_msg)

        start_ray_ip = f'{incoming_msg}:6379'
        log.info(
            f'trying to start ray on rank {node_rank} with ip: {start_ray_ip}',
        )
        cmd = [
            'ray',
            'start',
            f'--address={start_ray_ip}',
            '--resources={\"worker_node\": 8, \"accelerator_type:H100\":8}',
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                # capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # Note the errors here on MCLI are suppressed...
            log.error(f'error is: {e}')
            log.error(f'Command failed with exit code {e.returncode}')

        log.info(f'successfully started ray on node rank {node_rank}')
        dist.barrier()

    else:
        raise ValueError('NODE_RANK must be 0 or greater than 0')

    # dist.destroy_process_group()
    log.info('Finished initializing ray nodes.')

    if node_rank == 0:
        print('after destroy process group')
        for node in ray.nodes():
            log.info(f"Node: {node['NodeManagerAddress']}")
            log.info(f"Resources: {node['Resources']}")
            log.info(f"Alive: {node['Alive']}\n")
            # print(f"Node: {node['NodeManagerAddress']}")
            # print(f"Resources: {node['Resources']}")
            # print(f"Alive: {node['Alive']}\n")


def reassign_train_and_inference_ranks(num_train_nodes, num_inference_nodes):
    print("Master port is: ', os.getenv('MASTER_PORT'))")
    init_rank = int(os.getenv('NODE_RANK'))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE'))

    train_world_size = str(num_train_nodes * int(local_world_size))
    inf_world_size = str((num_inference_nodes) * int(local_world_size))

    if init_rank < num_train_nodes:
        os.environ['ORIGINAL_WORLD_SIZE'] = os.getenv('WORLD_SIZE')

        log.info('Reassinging env vars for training')

        # Duplicating some stuff here so we can retain this for rank 0
        os.environ['NUM_NODES'] = str(num_train_nodes)
        os.environ['TRAIN_NUM_NODES'] = str(num_train_nodes)

        os.environ['WORLD_SIZE'] = train_world_size
        os.environ['TRAIN_WORLD_SIZE'] = train_world_size

        if init_rank == 0:
            log.info(
                f'For node rank 0 setting world size to {num_inference_nodes} for ray.',
            )
            os.environ['NUM_NODES'] = str(num_inference_nodes)
            os.environ['WORLD_SIZE'] = str(inf_world_size)
            os.environ['TRAIN_MASTER_PORT'] = os.getenv('MASTER_PORT')
            # TODO: find a more stable way to find these ports...
            # the open port was found by socket bind...
            os.environ['MASTER_PORT'] = str(40977)

    else:
        log.info('Reassigning env vars for inference')
        os.environ['NODE_RANK'] = str(init_rank - num_train_nodes + 1)

        # We need to account for our master node here for communication
        os.environ['NUM_NODES'] = str(num_inference_nodes)
        os.environ['WORLD_SIZE'] = str(inf_world_size)
        os.environ['MASTER_PORT'] = str(40977)

        inf_node_rank = os.getenv('NODE_RANK')
        inf_world_size = os.getenv('WORLD_SIZE')
        inf_num_nodes = os.getenv('NUM_NODES')

        log.info(f'Node rank is: {inf_node_rank}')
        log.info(f'World size is: {inf_world_size}')
        log.info(f'num nodes is: {inf_num_nodes}')


if __name__ == '__main__':

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    with open(yaml_path) as f:
        yaml_cfg = om.load(f)

    num_nodes = os.getenv('NUM_NODES', None)
    assert num_nodes is not None, 'NUM_NODES must be set'
    num_nodes = int(num_nodes)

    num_train_nodes = yaml_cfg['variables']['num_train_nodes']
    # This includes the master node
    num_inference_nodes = num_nodes - num_train_nodes + 1

    reassign_train_and_inference_ranks(num_train_nodes, num_inference_nodes)

    start_ray_nodes()

    if os.getenv('NODE_RANK', None) == '0':
        os.environ['WORLD_SIZE'] = os.getenv('TRAIN_WORLD_SIZE')
        os.environ['NUM_NODES'] = os.getenv('TRAIN_NUM_NODES')
        os.environ['MASTER_PORT'] = os.getenv('TRAIN_MASTER_PORT')

    print('after start ray nodes')
    # Force flush logs for some reason??
    sys.stdout.flush()

    train_num_nodes = os.getenv('TRAIN_NUM_NODES', None)

    if train_num_nodes is not None:

        print("exiting on main training processes")
    else:
        # Have all inference nodes block until the training nodes are done
        # time.sleep(100000000)
        print("in inference node")

    # print ("sleeping for 3 minutes to make sure everything completes")
    time.sleep(10)
