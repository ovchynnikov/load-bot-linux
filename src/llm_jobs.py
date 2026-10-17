"""Bounded per-user LLM admission and shared best-effort typing indicators."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

from llm_delivery import message_topic_id


async def storage_call(function, *args, **kwargs):
    """Do not close SQLite while a cancelled coroutine's worker still uses it."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


@dataclass
class UserEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    references: int = 0


@dataclass
class JobTicket:
    user_id: int
    entry: UserEntry
    pending: bool = True
    released: bool = False
    started: bool = False


class QueueExpired(TimeoutError):
    """Only admission waits use this error, never provider/delivery deadlines."""


class JobGate:
    """Same-user waiters acquire their user lock before a global active slot."""

    def __init__(self, concurrency=2, pending=10, queue_timeout=120, jobs_per_user=3):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.pending_limit = pending
        self.queue_timeout = queue_timeout
        self.jobs_per_user = jobs_per_user
        self.pending_count = 0
        self.users = {}

    def admit(self, user_id):
        if self.pending_count >= self.pending_limit:
            return None
        entry = self.users.setdefault(user_id, UserEntry())
        if entry.references >= self.jobs_per_user:
            return None
        entry.references += 1
        self.pending_count += 1
        return JobTicket(user_id, entry)

    def release(self, ticket):
        if ticket.released:
            return
        ticket.released = True
        if ticket.pending:
            self.pending_count -= 1
        ticket.entry.references -= 1
        if ticket.entry.references == 0:
            del self.users[ticket.user_id]

    @asynccontextmanager
    async def slot(self, ticket):
        user_acquired = active_acquired = False
        try:
            try:
                async with asyncio.timeout(self.queue_timeout):
                    await ticket.entry.lock.acquire()
                    user_acquired = True
                    await self.semaphore.acquire()
                    active_acquired = True
            except TimeoutError as exc:
                raise QueueExpired from exc
            ticket.pending = False
            self.pending_count -= 1
            yield
        finally:
            if active_acquired:
                self.semaphore.release()
            if user_acquired:
                ticket.entry.lock.release()
            self.release(ticket)


@dataclass
class TypingTicket:
    key: tuple
    released: bool = False


class TypingIndicators:
    """One helper per chat/topic, even when multiple users have accepted jobs."""

    def __init__(self, interval=4, action_timeout=2):
        self.interval = interval
        self.action_timeout = action_timeout
        self.entries = {}
        self.tasks = set()

    def acquire(self, bot, message):
        key = (message.chat_id, message_topic_id(message), message.business_connection_id)
        if key in self.entries:
            self.entries[key][0] += 1
        else:
            task = asyncio.create_task(self._repeat(bot, key))
            self.entries[key] = [1, task]
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return TypingTicket(key)

    async def _repeat(self, bot, key):
        while True:
            started = asyncio.get_running_loop().time()
            with suppress(Exception):
                async with asyncio.timeout(self.action_timeout):
                    await bot.send_chat_action(
                        chat_id=key[0], action="typing", message_thread_id=key[1], business_connection_id=key[2]
                    )
            await asyncio.sleep(max(0, self.interval - (asyncio.get_running_loop().time() - started)))

    def discard(self, ticket):
        """Synchronous cleanup also works if a scheduled job never starts."""
        if ticket.released:
            return None
        ticket.released = True
        entry = self.entries.get(ticket.key)
        if entry is None:
            return None
        entry[0] -= 1
        if entry[0] == 0:
            del self.entries[ticket.key]
            entry[1].cancel()
            return entry[1]
        return None

    async def release(self, ticket):
        task = self.discard(ticket)
        if task:
            await asyncio.gather(task, return_exceptions=True)

    async def close(self):
        tasks = list(self.tasks)
        self.entries.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
