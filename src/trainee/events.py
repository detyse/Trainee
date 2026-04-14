from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import List

from trainee.models import EventMessage


class EventBus:
    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue[EventMessage]] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[EventMessage]:
        queue: asyncio.Queue[EventMessage] = asyncio.Queue(maxsize=64)
        async with self._lock:
            self._subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[EventMessage]) -> None:
        async with self._lock:
            with suppress(ValueError):
                self._subscribers.remove(queue)

    async def publish(self, event: EventMessage) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(event)
