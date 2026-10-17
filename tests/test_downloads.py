"""Download route: worker ownership, file-handle cleanup, timeouts and localized errors."""

import asyncio
import builtins
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from telegram.error import TelegramError

import downloads
import video_utils


def fake_update(text='https://tiktok.com/@u/video/1'):
    chat = SimpleNamespace(send_video=AsyncMock(), send_photo=AsyncMock(), send_media_group=AsyncMock())
    message = SimpleNamespace(text=text, message_id=5, entities=None, chat=chat, reply_text=AsyncMock())
    return SimpleNamespace(message=message, effective_chat=SimpleNamespace(type='private', id=1))


class RecordingOpen:
    """Wrap builtins.open so tests can check that every opened file got closed."""

    def __init__(self):
        self.files = []

    def __call__(self, path, mode='rb'):
        assert mode == 'rb', mode  # the download route only opens media in binary mode
        handle = builtins.open(path, 'rb')
        self.files.append(handle)
        return handle


class RunBlockingTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_loop_keeps_running_during_worker(self):
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        await downloads.run_blocking(time.sleep, 0.2)
        task.cancel()
        self.assertGreater(len(ticks), 5)

    async def test_cancelled_download_result_is_discarded(self):
        release, discarded = threading.Event(), []
        task = asyncio.create_task(
            downloads.run_blocking(lambda: release.wait(5) and '/tmp/x', discard=discarded.append)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        asyncio.get_running_loop().call_later(0.05, release.set)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(discarded, ['/tmp/x'])

    async def test_worker_error_during_cancellation_still_cancels(self):
        release, discarded = threading.Event(), []

        def fail():
            release.wait(5)
            raise RuntimeError('worker failed')

        task = asyncio.create_task(downloads.run_blocking(fail, discard=discarded.append))
        await asyncio.sleep(0.05)
        task.cancel()
        asyncio.get_running_loop().call_later(0.05, release.set)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(discarded, [])

    async def test_cleanup_waits_for_cancelled_compression(self):
        started, release, order = threading.Event(), threading.Event(), []

        def compress(_path):
            started.set()
            release.wait(5)
            order.append('compressed')

        with (
            patch.object(downloads, 'is_video_too_long_to_download', return_value=False),
            patch.object(downloads, 'download_media', return_value='/tmp/video.mp4'),
            patch.object(downloads, 'is_video_duration_over_limits', return_value=False),
            patch.object(downloads, 'is_large_file', return_value=True),
            patch.object(downloads, 'compress_video', side_effect=compress),
            patch.object(downloads, 'cleanup', side_effect=lambda _paths: order.append('cleanup')),
        ):
            task = asyncio.create_task(downloads.handle_download(fake_update()))
            await asyncio.to_thread(started.wait, 5)
            task.cancel()
            asyncio.get_running_loop().call_later(0.05, release.set)
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(order, ['compressed', 'cleanup'])


class RepeatedCancellationTests(unittest.IsolatedAsyncioTestCase):
    """A second cancel while draining must not abandon the worker (R2-F1)."""

    async def cancel_twice_then_release(self, task, started, release):
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertFalse(task.done(), 'handler exited while the worker was still running')
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_download_stage(self):
        started, release, order = threading.Event(), threading.Event(), []

        def download(_url):
            started.set()
            release.wait(5)
            order.append('downloaded')
            return '/tmp/downloaded.mp4'

        with (
            patch.object(downloads, 'is_video_too_long_to_download', return_value=False),
            patch.object(downloads, 'download_media', side_effect=download),
            patch.object(downloads, 'cleanup', side_effect=lambda paths: order.append(('cleanup', paths))),
        ):
            task = asyncio.create_task(downloads.handle_download(fake_update()))
            await self.cancel_twice_then_release(task, started, release)
        self.assertEqual(order, ['downloaded', ('cleanup', '/tmp/downloaded.mp4')])

    async def test_compression_stage(self):
        started, release, order = threading.Event(), threading.Event(), []

        def compress(_path):
            started.set()
            release.wait(5)
            order.append('compressed')

        with (
            patch.object(downloads, 'is_video_too_long_to_download', return_value=False),
            patch.object(downloads, 'download_media', return_value='/tmp/video.mp4'),
            patch.object(downloads, 'is_video_duration_over_limits', return_value=False),
            patch.object(downloads, 'is_large_file', return_value=True),
            patch.object(downloads, 'compress_video', side_effect=compress),
            patch.object(downloads, 'cleanup', side_effect=lambda _paths: order.append('cleanup')),
        ):
            task = asyncio.create_task(downloads.handle_download(fake_update()))
            await self.cancel_twice_then_release(task, started, release)
        self.assertEqual(order, ['compressed', 'cleanup'])

    async def test_worker_error(self):
        started, release, discarded = threading.Event(), threading.Event(), []

        def fail():
            started.set()
            release.wait(5)
            raise RuntimeError('worker failed')

        task = asyncio.create_task(downloads.run_blocking(fail, discard=discarded.append))
        await self.cancel_twice_then_release(task, started, release)
        self.assertEqual(discarded, [])


class SendTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_missing_file_raises_real_error(self):
        with patch.object(downloads, 'get_video_dimensions', return_value=(None, None)):
            with self.assertRaises(FileNotFoundError):
                await downloads.send_video(fake_update(), '/nonexistent/video.mp4', False)
            with self.assertRaises(FileNotFoundError):
                await downloads.send_pic(fake_update(), '/nonexistent/pic.jpg')

    async def test_group_open_or_probe_failure_closes_earlier_files(self):
        with TemporaryDirectory() as temp:
            first = Path(temp, 'a.mp4')
            first.write_bytes(b'x')
            missing = str(Path(temp, 'missing.mp4'))
            for call, items in (
                (lambda update, items: downloads.send_video(update, items, False), [str(first), missing]),
                (downloads.send_pic, [str(first), missing]),
            ):
                recorder = RecordingOpen()
                with (
                    patch.object(downloads, 'open', recorder, create=True),
                    patch.object(downloads, 'get_video_dimensions', return_value=(1, 1)),
                ):
                    with self.assertRaises(FileNotFoundError):
                        await call(fake_update(), items)
                self.assertTrue(recorder.files and all(handle.closed for handle in recorder.files))

            recorder = RecordingOpen()
            with (
                patch.object(downloads, 'open', recorder, create=True),
                patch.object(downloads, 'get_video_dimensions', side_effect=[(1, 1), RuntimeError('probe')]),
            ):
                with self.assertRaises(RuntimeError):
                    await downloads.send_video(fake_update(), [str(first), str(first)], False)
            self.assertTrue(all(handle.closed for handle in recorder.files))

    async def test_group_success_sends_and_closes(self):
        with TemporaryDirectory() as temp:
            files = [Path(temp, f'{index}.jpg') for index in range(3)]
            for file in files:
                file.write_bytes(b'x')
            recorder = RecordingOpen()
            update = fake_update()
            with patch.object(downloads, 'open', recorder, create=True):
                await downloads.send_pic(update, [str(file) for file in files])
            self.assertEqual(len(update.message.chat.send_media_group.await_args.kwargs['media']), 3)
            self.assertTrue(all(handle.closed for handle in recorder.files))

    async def test_telegram_error_reply_is_localized_without_raw_text(self):
        with TemporaryDirectory() as temp:
            video = Path(temp, 'v.mp4')
            video.write_bytes(b'x')
            for language, expected in (('uk', 'Не вдалося надіслати відео'), ('en', "Couldn't send the video")):
                update = fake_update()
                update.message.chat.send_video.side_effect = TelegramError('PRIVATE_RAW_TEXT')
                with (
                    patch.object(downloads, 'language', language),
                    patch.object(downloads, 'get_video_dimensions', return_value=(1, 1)),
                ):
                    await downloads.send_video(update, str(video), False)
                reply = update.message.reply_text.await_args.args[0]
                self.assertIn(expected, reply)
                self.assertNotIn('PRIVATE_RAW_TEXT', reply)

    async def test_too_long_reply_is_localized(self):
        for language, expected in (('uk', 'Відео задовге'), ('en', 'The video is too long')):
            update = fake_update()
            with (
                patch.object(downloads, 'language', language),
                patch.object(downloads, 'is_video_too_long_to_download', return_value=True),
            ):
                await downloads.handle_download(update)
            self.assertIn(expected, update.message.reply_text.await_args.args[0])


class VideoUtilsTimeoutTests(unittest.TestCase):
    def test_ffprobe_timeout_returns_unknown(self):
        with patch.object(video_utils.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffprobe', 30)) as run:
            self.assertIsNone(video_utils.get_video_duration('/tmp/v.mp4'))
            self.assertEqual(video_utils.get_video_dimensions('/tmp/v.mp4'), (None, None))
        self.assertEqual(run.call_args.kwargs['timeout'], video_utils.FFPROBE_TIMEOUT_SEC)

    def test_compression_timeout_keeps_original_and_removes_partial_output(self):
        with TemporaryDirectory() as temp:
            source = Path(temp, 'v.mp4')
            source.write_bytes(b'original')
            outputs = []

            def run(command, **kwargs):
                outputs.append(command[-1])
                Path(command[-1]).write_bytes(b'partial')
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])

            with (
                patch.object(video_utils, 'get_video_duration', return_value=10.0),
                patch.object(video_utils.subprocess, 'run', side_effect=run) as fake_run,
            ):
                video_utils.compress_video(str(source))
            self.assertEqual(fake_run.call_args.kwargs['timeout'], video_utils.FFMPEG_TIMEOUT_SEC)
            self.assertEqual(source.read_bytes(), b'original')
            self.assertFalse(Path(outputs[0]).exists())

    def test_metadata_uses_socket_timeout(self):
        captured = {}

        class FakeYDL:
            def __init__(self, options):
                captured.update(options)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, _url, download=False):  # pylint: disable=unused-argument
                return {'duration': 1}

        with patch.object(video_utils.yt_dlp, 'YoutubeDL', FakeYDL):
            video_utils.get_video_metadata('https://tiktok.com/@u/video/1')
        self.assertEqual(captured['socket_timeout'], video_utils.METADATA_SOCKET_TIMEOUT_SEC)


if __name__ == '__main__':
    unittest.main()
