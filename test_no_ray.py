import subprocess
import traceback


if __name__ == "__main__":
    # test on 4 gpus!
    # for multinode, we should determine which command to launch on which node
    try:
        # Launch the train agent with multiple processes for distributed training
        p1 = subprocess.Popen('CUDA_VISIBLE_DEVICES=0,1 composer -n 2 train.py', shell=True)
        
        # Launch the rollout agent with a single process for vllm
        p2 = subprocess.Popen('CUDA_VISIBLE_DEVICES=2,3 python rollout.py', shell=True)
        p1.wait()
        p2.wait()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print('Killing training processes')
    finally:
        p1.terminate()
        p2.terminate()

