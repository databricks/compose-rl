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
from llmfoundry.command_utils import train_from_yaml
from omegaconf import OmegaConf as om

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@ray.remote
class SyncActor:
    def __init__(self):
         self.training_done = False

    def mark_done(self):
         self.training_done = True

    def wait_for_training(self):
         while not self.training_done:
             time.sleep(10)
         return "Training complete!"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Handles both color codes and formatting like bold, underline, etc.

    Args:
        text (str): The input string potentially containing ANSI codes.
    """
    ansi_pattern = r'\x1b\[[0-9;]*[a-zA-Z]'
    return re.sub(ansi_pattern, '', text)


def parse_ray_local_ip(log_output: str) -> str:
    """Parse the local node IP from Ray runtime startup logs.

    Args:
        log_output (str): The complete log output from Ray startup

    Returns:
        str: The extracted local node IP address
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
    """Broadcast a string message from the source rank to all other ranks.

    Args:
        message (str): The message to broadcast
        src_rank (int): The rank of the source process
    """
    encoded = message.encode('utf-8')
    length_tensor = torch.LongTensor([len(encoded)])
    dist.broadcast(length_tensor, src=src_rank)

    data_tensor = torch.ByteTensor(list(encoded)) if dist.get_rank() == src_rank else \
                  torch.ByteTensor([0] * length_tensor.item()) # type: ignore
    dist.broadcast(data_tensor, src=src_rank)

    return data_tensor.cpu().numpy().tobytes().decode('utf-8')


def recv_string(src: int) -> str:
    """Receive the length of a string, then receive the actual bytes and decode.

    Args:
        src (int): The rank of the source process
    """
    length_tensor = torch.LongTensor([0])
    dist.recv(length_tensor, src=src)

    data_tensor = torch.ByteTensor(length_tensor.item())
    dist.recv(data_tensor, src=src)

    return data_tensor.numpy().tobytes().decode('utf-8')


def start_ray_nodes():
    rank = int(os.getenv('NODE_RANK'))  # type: ignore
    world_size = int(os.getenv('NUM_NODES'))  # type: ignore
    local_rank = os.getenv('LOCAL_RANK', None)
    assert local_rank is not None, "LOCAL_RANK is usually set via composer"
    local_rank = int(local_rank)

    train_num_nodes = os.getenv('TRAIN_NUM_NODES', None)

    if train_num_nodes is not None and rank != 0:
        log.info(
            "On a training node or rank that isn't the master node no need to start ray.",
        )
        return
    
    if local_rank != 0:
        log.info('Not starting ray on non-master local rank, exiting.')
        return

    vars_to_check = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'NODE_RANK']
    for var in vars_to_check:
        log.warning(f"{var}: {os.environ.get(var, 'Not set')}")

    log.info('Starting gloo backend process.')

    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    log.info('Finished setting up gloo backend process.')

    node_rank = os.getenv('NODE_RANK', None)
    if node_rank is None:
        raise ValueError('NODE_RANK must be set')
    node_rank = int(node_rank)

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

    elif node_rank > 0:
        # Ranks 1..(world_size-1) -> receive message from rank 0
        incoming_msg = broadcast_string('', src_rank=0)
        log.info(f'[Rank {rank}] Received message from rank 0: {incoming_msg}')

        start_ray_ip = f'{incoming_msg}:6379'
        log.info(
            f'trying to start ray on rank {node_rank} with ip: {start_ray_ip}',
        )

        # We use worker node as a variable to consume later
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
            log.error(f'error is: {e}')
            log.error(f'Command failed with exit code {e.returncode}')

        log.info(f'successfully started ray on node rank {node_rank}')
        dist.barrier()

    else:
        raise ValueError('NODE_RANK must be 0 or greater than 0')

    log.info('Starting to destroy process group.')
    dist.destroy_process_group()
    log.info('Finished destroying process group.')
    log.info('Finished initializing ray nodes.')


def reassign_train_and_inference_ranks(
    num_train_nodes: int,
    num_inference_nodes: int,
):
    """Reassigns the ranks for training and inference nodes.

    Args:
        num_train_nodes (int): The number of training nodes
        num_inference_nodes (int): The number of inference nodes
    """
    node_rank = os.getenv('NODE_RANK', None)
    local_world_size = os.getenv('LOCAL_WORLD_SIZE', None)
    assert node_rank is not None
    assert local_world_size is not None
    init_rank = int(node_rank)
    local_world_size = int(local_world_size)
    
    local_rank = os.getenv('LOCAL_RANK', None)
    assert local_rank is not None, "LOCAL_RANK is usually set via composer"
    local_rank = int(local_rank)    

    train_world_size = str(num_train_nodes * int(local_world_size))

    if init_rank < num_train_nodes:
        log.info('Reassinging env vars for training')
        world_size = os.getenv('WORLD_SIZE', None)
        master_port = os.getenv('MASTER_PORT', None)
        assert world_size is not None
        assert master_port is not None

        os.environ['NUM_NODES'] = str(num_train_nodes)
        os.environ['TRAIN_NUM_NODES'] = str(num_train_nodes)

        os.environ['WORLD_SIZE'] = train_world_size
        os.environ['TRAIN_WORLD_SIZE'] = train_world_size

        if init_rank == 0 and local_rank == 0:
            log.info(
                f'For node 0 and rank 0 setting world size to {num_inference_nodes} to set up ray.',
            )
            os.environ['NUM_NODES'] = str(num_inference_nodes)
            # Need to set this here to avoid duplication 
            os.environ['TRAIN_MASTER_PORT'] = master_port
            # TODO: find a more stable way to find these ports.
            # the open port was found by socket bind...
            os.environ['MASTER_PORT'] = str(40977)

    else:
        log.info('Reassigning env vars for inference')
        os.environ['NODE_RANK'] = str(init_rank - num_train_nodes + 1)

        # We need to account for our master node here for communication
        os.environ['NUM_NODES'] = str(num_inference_nodes)
        os.environ['MASTER_PORT'] = str(40977)


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    with open(yaml_path) as f:
        yaml_cfg = om.load(f)

    num_nodes = os.getenv('NUM_NODES', None)
    assert num_nodes is not None, 'NUM_NODES must be set'
    num_nodes = int(num_nodes)

    num_train_nodes = yaml_cfg['variables']['num_train_nodes']  # type: ignore
    # This includes the master node
    num_inference_nodes = num_nodes - num_train_nodes + 1

    reassign_train_and_inference_ranks(num_train_nodes, num_inference_nodes)

    start_ray_nodes()

    if os.getenv('NODE_RANK', None) == '0' and os.getenv('LOCAL_RANK', None) == '0':
        train_world_size = os.getenv('TRAIN_WORLD_SIZE', None)
        train_num_nodes = os.getenv('TRAIN_NUM_NODES', None)
        master_port = os.getenv('TRAIN_MASTER_PORT', None)

        assert train_world_size is not None
        assert train_num_nodes is not None
        assert master_port is not None

        os.environ['WORLD_SIZE'] = train_world_size
        os.environ['NUM_NODES'] = train_num_nodes
        os.environ['MASTER_PORT'] = master_port

        # Adding a ray sync actor on global rank 0 to amek it work
        sync_actor = SyncActor.remote()
        ray.util.register_actor("sync_actor", sync_actor)

    log.info('after start ray nodes')

    train_num_nodes = os.getenv('TRAIN_NUM_NODES', None)

    if train_num_nodes is not None:
        # log.info('exiting on main training processes')
        train_from_yaml(yaml_path, args_list)
        ray.get(sync_actor.mark_done.remote())
    else:
        # Have all inference nodes block until the training nodes are done
        log.info('in inference node')
        log.info('setting up ray sync actor')
        sync_actor = ray.get_actor('sync_actor')
        result = ray.get(sync_actor.wait_for_training.remote())
