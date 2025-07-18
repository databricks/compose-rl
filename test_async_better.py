import asyncio
import time

program_start_time = time.time()

async def async_function(name: str):
    print(f"Async function {name} started at {time.time() - program_start_time:.2f}s")
    await asyncio.sleep(1)
    print(f"Async function {name} finished at {time.time() - program_start_time:.2f}s")

# Better semaphore pattern
total_tasks = 4
concurrent_tasks = 2
semaphore = asyncio.Semaphore(concurrent_tasks)

async def async_function_with_semaphore_better(name: str):
    async with semaphore:  # This acquires and releases automatically
        await async_function(name)

# Alternative explicit version:
async def async_function_with_semaphore_explicit(name: str):
    await semaphore.acquire()
    try:
        await async_function(name)
    finally:
        semaphore.release()

async def run_tasks_better():
    # Create all tasks at once - semaphore limiting happens inside each task
    tasks = [
        asyncio.create_task(async_function_with_semaphore_better(f"Task {i}"))
        for i in range(total_tasks)
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    print("Running better pattern:")
    asyncio.run(run_tasks_better()) 