import asyncio
from SLOsServe.router.sem_util import MaxCapSemaphore
import time 
sem = MaxCapSemaphore(10)
Q = []
async def router():
    while True: 
        await asyncio.sleep(0)
        await sem.reset()
        print(f'process {len(Q)}')  
        Q.clear()
        
async def add_req(id: int):
    await sem.acquire()
    Q.append(id)
    

async def main():
    router_task = asyncio.create_task(router())

    add_req_tasks = [add_req(i) for i in range(100)]     

    await asyncio.gather(*add_req_tasks)
    router_task.cancel()
    try: 
        await router_task    
    except asyncio.CancelledError:
        pass 
    

asyncio.run(main())