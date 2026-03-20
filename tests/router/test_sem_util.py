import asyncio

import pytest

from SLOsServe.router.sem_util import RefillableSemaphore


@pytest.mark.asyncio
async def test_refillable_semaphore_waits_for_refill():
    sem = RefillableSemaphore(initial_value=1)

    await sem.acquire()
    assert sem.locked()

    acquired = False

    async def waiter():
        nonlocal acquired
        await sem.acquire()
        acquired = True

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)

    assert not acquired

    added = await sem.refill(2)

    assert added == 2
    await asyncio.wait_for(task, timeout=0.1)
    assert acquired
    assert sem.value == 1


@pytest.mark.asyncio
async def test_refillable_semaphore_caps_refill_at_max_value():
    sem = RefillableSemaphore(initial_value=1, max_value=3)

    added = await sem.refill(5)

    assert added == 2
    assert sem.value == 3


@pytest.mark.asyncio
async def test_refillable_semaphore_try_acquire_reports_exhaustion():
    sem = RefillableSemaphore(initial_value=1)

    assert await sem.try_acquire()
    assert not await sem.try_acquire()
    assert sem.value == 0


@pytest.mark.asyncio
async def test_refillable_semaphore_release_aliases_refill():
    sem = RefillableSemaphore(initial_value=0)

    added = await sem.release(3)

    assert added == 3
    assert sem.value == 3


def test_refillable_semaphore_validates_inputs():
    with pytest.raises(ValueError):
        RefillableSemaphore(initial_value=-1)
    with pytest.raises(ValueError):
        RefillableSemaphore(initial_value=2, max_value=1)

    sem = RefillableSemaphore(initial_value=1)
    with pytest.raises(ValueError):
        asyncio.run(sem.refill(0))
