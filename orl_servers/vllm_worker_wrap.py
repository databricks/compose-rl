import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


def stateless_init_process_group(master_address: str, master_port: str, rank: int, world_size: int, device: str):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerWrap:

    _model_update_group: PyNcclCommunicator | None = None

    def init_process_group(
        self, master_address: str, master_port: str, rank_offset: int, world_size: int
    ):
        """Init torch process group for model weights update"""

        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = stateless_init_process_group(
            master_address,
            int(master_port),
            rank,
            world_size,
            self.device,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}",
        )

    def update_weight(self, name: str, dtype: torch.dtype, shape: tuple[int, ...] | list[int], empty_cache: bool = False):

        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()