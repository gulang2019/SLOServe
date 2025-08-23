from typing import List, Callable, Dict, Tuple
from .object import ObjectRef

def comm_wrapper(*args, **kwargs):
    def impl(
        comm_impl,
        device_group,
        communicator,   
    ):
        comm_impl(device_group, communicator, *args, **kwargs)
    return impl 

class Communicator:
    def __init__(self):
        self.jobs: Dict[Tuple[int, int], List[Tuple[ObjectRef, bool]]] = {}
        
    def broadcast(self, src: int, dsts: List[int], obj: ObjectRef, keep_old: bool = False):
        for dst in dsts:
            self.comm(src, dst, obj, keep_old)
    
    def comm(self, src: int, dst: int, obj: ObjectRef, keep_old: bool = False):
        if src == dst: return
        if (src, dst) not in self.jobs:
            self.jobs[(src, dst)] = ([], [])
        jobs = self.jobs[(src, dst)]
        jobs[0].append(obj)
        jobs[1].append(keep_old)

    def batched_commit(self, comm_impl: Callable[[int, int, ObjectRef], None]):
        for (src, dst), refs in self.jobs.items():
            comm_impl(src, dst, *refs)
        self.jobs = {}

    async def batched_commit_async(self, comm_impl: Callable[[int, int, ObjectRef], None]):
        for (src, dst), refs in self.jobs.items():
            await comm_impl(src, dst, *refs)
        self.jobs = {}