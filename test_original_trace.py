import asyncio
import time

program_start_time = time.time()

def log(message: str):
    print(f"{time.time() - program_start_time:.3f}s: {message}")

async def async_function(name: str):
    log(f"  → {name} STARTED")
    await asyncio.sleep(1)
    log(f"  ← {name} FINISHED")

total_tasks = 4
concurrent_tasks = 2
semaphore = asyncio.Semaphore(concurrent_tasks)

async def async_function_with_semaphore(name: str):
    cor = async_function(name)
    await asyncio.sleep(1)
    await cor
    log(f"  {name} calling semaphore.release()")
    semaphore.release()

async def run_tasks_with_trace():
    log("Starting run_tasks()")
    tasks = []
    
    for i in range(total_tasks):
        log(f"Iteration {i}: About to acquire semaphore")
        await semaphore.acquire()  # This is where the magic happens!
        log(f"Iteration {i}: Acquired semaphore, creating task")
        tasks.append(asyncio.create_task(async_function_with_semaphore(f"Task {i}")))
        log(f"Iteration {i}: Task created")
    
    log("All tasks created, calling gather()")
    await asyncio.gather(*tasks)
    log("gather() completed")

if __name__ == "__main__":
    asyncio.run(run_tasks_with_trace()) 