import ray
import subprocess


def ray_start():
    subprocess.run(['ray', 'start', '--head', '--port=6379'], check=True)


def ray_stop():
    subprocess.run(['ray', 'stop'], check=True)

def ray_init():
    ray.init()


def ray_shutdown():
    ray.shutdown()

if __name__ == '__main__':
    ray_start()
    ray_init()
    ray_shutdown()
    ray_stop()
