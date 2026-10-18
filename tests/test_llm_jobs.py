"""Concurrency, cancellation and typing lifecycle checks without Telegram access."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from db_storage import BotStorage
from grok_api import GrokService
from llm_config import LLMSettings
from llm_jobs import JobGate, QueueExpired, TypingIndicators, storage_call


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_and_other_users_progress_without_a_worker_pool(self):
        gate = JobGate(2, 10)
        started, release = asyncio.Event(), asyncio.Event()
        events = []

        async def job(user, index):
            ticket = gate.admit(user)
            async with gate.slot(ticket):
                events.append(index)
                if index == 1:
                    started.set()
                    await release.wait()

        first = asyncio.create_task(job(1, 1))
        await started.wait()
        second = asyncio.create_task(job(1, 2))
        other = asyncio.create_task(job(2, 3))
        await other
        self.assertEqual(events, [1, 3])
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(events, [1, 3, 2])
        self.assertEqual(gate.users, {})
        self.assertEqual(gate.pending_count, 0)

    async def test_queue_refusal_expiry_cancel_and_lock_handoff(self):
        gate = JobGate(1, 1, 0.01)
        ticket = gate.admit(1)
        self.assertIsNone(gate.admit(2))
        async with gate.slot(ticket):
            waiter = gate.admit(1)
            self.assertIs(waiter.entry, ticket.entry)
            with self.assertRaises(QueueExpired):
                async with gate.slot(waiter):
                    self.fail('Expired job ran')
            self.assertEqual(ticket.entry.references, 1)
            cancelled = gate.admit(2)
            job = asyncio.create_task(self.acquire(gate, cancelled))
            await asyncio.sleep(0)
            job.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await job
            self.assertEqual(gate.pending_count, 0)
        self.assertEqual(gate.users, {})
        async with gate.slot(gate.admit(1)):
            self.assertEqual(gate.pending_count, 0)

    async def acquire(self, gate, ticket):
        async with gate.slot(ticket):
            await asyncio.sleep(10)

    async def test_body_timeout_is_not_mislabeled_queue_expiry(self):
        gate = JobGate()
        with self.assertRaises(TimeoutError) as caught:
            async with gate.slot(gate.admit(1)):
                raise TimeoutError('provider')
        self.assertNotIsInstance(caught.exception, QueueExpired)
        self.assertFalse(gate.users)

    async def test_storage_worker_is_joined_on_cancellation(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def worker():
            started.set()
            release.wait(1)
            finished.set()

        task = asyncio.create_task(storage_call(worker))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())

    async def test_cancel_during_reservation_refunds_before_dispatch(self):
        with TemporaryDirectory() as temp:
            db = BotStorage(str(Path(temp) / 'test.db'))
            started, release = threading.Event(), threading.Event()
            original = db.reserve_search

            def slow(*args):
                result = original(*args)
                started.set()
                release.wait(1)
                return result

            db.reserve_search = slow
            client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock()))
            service = GrokService(client, db, LLMSettings(search_enabled=True))
            task = asyncio.create_task(service.generate([], 1, True))
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(client.responses.create.await_count, 0)
            self.assertIsNotNone(original(1, 1, 1))
            db.close()

    async def test_typing_is_shared_best_effort_and_stops_at_last_release(self):
        typing = TypingIndicators(interval=0.005, action_timeout=0.005)
        bot = SimpleNamespace(send_chat_action=AsyncMock(side_effect=RuntimeError('action failure')))
        message = SimpleNamespace(chat_id=1, message_thread_id=2, is_topic_message=True, business_connection_id=None)
        first = typing.acquire(bot, message)
        second = typing.acquire(bot, message)
        self.assertEqual(len(typing.entries), 1)
        await asyncio.sleep(0.015)
        self.assertGreaterEqual(bot.send_chat_action.await_count, 1)
        await typing.release(first)
        self.assertTrue(typing.entries)
        await typing.release(second)
        count = bot.send_chat_action.await_count
        await asyncio.sleep(0.015)
        self.assertEqual(bot.send_chat_action.await_count, count)
        self.assertFalse(typing.entries)
        typing.acquire(bot, message)
        await typing.close()
        self.assertFalse(typing.entries)

    async def test_per_user_cap_counts_active_and_pending_and_releases_on_cancellation(self):
        gate = JobGate(concurrency=2, pending=10, jobs_per_user=3)
        active = gate.admit(1)
        async with gate.slot(active):
            waiters = [gate.admit(1), gate.admit(1)]
            self.assertIsNone(gate.admit(1))
            other = gate.admit(2)
            async with gate.slot(other):
                self.assertEqual(active.entry.references, 3)
            # Simulate a task cancelled before its first coroutine step.
            gate.release(waiters[0])
            gate.release(waiters[0])
            replacement = gate.admit(1)
            self.assertIsNotNone(replacement)
            # An admitted waiter cancelled while blocked must also free its slot.
            waiting = asyncio.create_task(self.acquire(gate, waiters[1]))
            await asyncio.sleep(0)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            gate.release(replacement)
            self.assertEqual(active.entry.references, 1)
        self.assertFalse(gate.users)
        self.assertEqual(gate.pending_count, 0)
        # Cancellation of an active job releases user admission as well.
        task = asyncio.create_task(self.acquire(gate, gate.admit(1)))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(gate.users)
