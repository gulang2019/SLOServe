import asyncio


class MaxCapSemaphore:

    def __init__(self, capacity: int = 1):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        
        self._capacity = capacity
        self._value = capacity
        self._cond = asyncio.Condition()

    @property
    def value(self) -> int:
        return self._value
    
    async def acquire(self) -> None:
        async with self._cond:
            while self._value == 0:
                await self._cond.wait()
            self._value -= 1

    async def reset(self) -> int:
        async with self._cond:
            previous = self._value
            self._value = self._capacity 
            added = self._value - previous
            if added > 0:
                self._cond.notify_all()
            return added