from typing import List, Tuple, Union, Iterable
from dataclasses import dataclass, field 

ALL_DEVICE_GROUPS = dict()

def get_device_group(devices: Union[List[int], int]):
    if isinstance(devices, int):
        devices = [devices]
    key = tuple(sorted(devices))
    if key not in ALL_DEVICE_GROUPS: 
        ALL_DEVICE_GROUPS[key] = DeviceGroup(key)
    return ALL_DEVICE_GROUPS[key]

@dataclass 
class DeviceGroup:
    devices: Tuple = tuple()

    def __iter__(self):
        return iter(self.devices)

    def __hash__(self):
        return hash(self.devices)

    def __len__(self):
        return len(self.devices)

    def __copy__(self):
        raise RuntimeError

    def __deepcopy__(self):
        raise RuntimeError
    
    def __repr__(self) -> str:
        return f'DeviceGroup{self.devices}'

class Placement: 
    pattern: Tuple[int, int]
    device_groups: List[DeviceGroup]

    def __getitem__(self, key) -> Tuple[int, ...]:
        if isinstance(key, int):
            return self.device_groups[key].devices 
        elif isinstance(key, tuple):
            assert isinstance(key[0], int)
            device_group = self.device_groups[key[0]]
            ret = []
            def impl(i, idx):
                if i == 2: 
                    ret.append(device_group.devices[idx])
                    return 
                _key = key[i+1] if len(key) > (i+1) else slice(None, None, 1)
                step = (self.pattern[-1] if i == 0 else 1)
                if isinstance(_key, int):
                    if _key < 0: _key += self.pattern[i]
                    impl(i+1, idx + _key * step)
                elif isinstance(_key, slice):
                    assert _key.start == None and _key.stop == None and (_key.step == 1 or _key.step is None)
                    for j in range(self.pattern[i]):
                        impl(i+1, idx + j * step)
            impl(0, 0)
            return tuple(ret) 
        else: raise TypeError

    def __len__(self):
        return self.length

    def __init__(self,
                 x: int = 1, 
                 y: int = 1):
        self.pattern = (x,y)
        self.length = x * y
        self.device_groups = []

    def add_device_group(self, device_group) -> 'Placement':
        if isinstance(device_group, DeviceGroup):
            self.device_groups.append(device_group)
        elif isinstance(device_group, int):
            self.device_groups.append(get_device_group([device_group]))
        else:
            device_ids = list(device_group)
            assert self.length == len(device_ids)
            self.device_groups.append(get_device_group(device_ids))    
        return self
    
    def update_device_group(self, device_group) -> 'Placement':
        self.device_groups = []
        self.add_device_group(device_group)
        return self 

    def add_device_groups(self, iterable) -> 'Placement':
        for device_group in iterable:
            self.add_device_group(device_group)
        return self 

    def update_device_groups(self, iterable)-> 'Placement':
        self.device_groups = []
        self.add_device_groups(iterable)
        return self 

    def __contains__(self, key):
        return get_device_group(key) in self.device_groups

    def __repr__(self):
        return f'Placement({self.pattern}: ' + ','.join(map(str, self.device_groups)) + ')'

@dataclass 
class ClusterStatus:
    n_device: int 
    memory_bandwidth: float = field(init = False) 
    loads: List[float] = field(init = False)
    max_load_gap: float = 1e9
    
    def get_min_load_device_group(self, device_groups: List[Iterable]):
        return min(device_groups, key = lambda x: min(map(self.loads.__getitem__, x)))

    def add_load(self, device_id: int, load: float):
        self.loads[device_id] += load

    def get_max_load(self, devices: Iterable):
        return max(map(self.loads.__getitem__, devices))

    def __post_init__(self):
        from .utils import profile_memory_bandwidth
        self.memory_bandwidth = profile_memory_bandwidth() 
        self.loads = [0 for _ in range(self.n_device)]