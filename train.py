from composer.utils import dist

from multiprocessing.context import SpawnProcess

def train_to_inference_communication():
    pass

if __name__ == "__main__":
    dist.initialize_dist()
    print(f"Hello from rank {dist.get_rank()}")

    SpawnProcess(target=train_to_inference_communication).start()