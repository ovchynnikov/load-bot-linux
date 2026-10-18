"""Import/start the real bot in a separate process with dummy credentials and temp data."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class BotIntegrationTests(unittest.TestCase):
    def test_startup_routing_generation_delivery_and_cleanup(self):
        root = Path(__file__).resolve().parents[1]
        script = r'''
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import main
import downloads
from grok_api import GrokResult
from llm_text import localized

class App:
    def __init__(self):
        self.tasks = []
    def create_task(self, coroutine, **kwargs):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

def update(text, user=1):
    message = SimpleNamespace(text=text, message_id=17, message_thread_id=3, is_topic_message=False,
        business_connection_id=None, chat_id=99, reply_text=AsyncMock())
    return SimpleNamespace(message=message, effective_user=SimpleNamespace(id=user, username='tester'),
        effective_chat=SimpleNamespace(id=99, type='private'))

async def check():
    app = App()
    context = SimpleNamespace(application=app, bot=SimpleNamespace(send_chat_action=AsyncMock()))
    main.grok_service.generate = AsyncMock(return_value=GrokResult('Answer[[1]](https://example.org/a)', 'used'))
    first = update('Ботяра, web: https://Host/AbC?Key=X and "ботяра image: cat"')
    await main.handle_message(first, context)
    await asyncio.gather(*app.tasks)
    args = main.grok_service.generate.call_args
    assert args.args[0][-1]['content'].startswith('https://Host/AbC?Key=X')
    assert args.kwargs['explicit'] is True
    assert localized('sources', main.language) in first.message.reply_text.call_args.args[0]
    assert first.message.reply_text.call_args.kwargs['reply_to_message_id'] == 17
    assert not main.llm_gate.users and not main.typing_indicators.entries
    assert main.db_storage.load_user_data(1)['daily_count'] == 1
    assert 'https://example.org/a' not in main.conversation_context[1][-1][1]
    second = update('ботяра, second')
    second.message.reply_text.side_effect = RuntimeError('PRIVATE_DELIVERY_SENTINEL')
    await main.schedule_llm_message(second, context)
    await asyncio.gather(*app.tasks)
    assert main.grok_service.generate.await_count == 2
    assert main.db_storage.load_user_data(1)['daily_count'] == 2
    messages = main.grok_service.generate.call_args.args[0]
    assert [item['role'] for item in messages] == ['system', 'user', 'assistant', 'user']

    main.grok_service.generate.return_value = GrokResult('No search', notice='quota', notice_day=main.db_storage.utc_day())
    third = update('ботяра, quota', 2)
    await main.schedule_llm_message(third, context)
    await asyncio.gather(*app.tasks)
    assert main.db_storage.search_notice_shown(2, main.db_storage.utc_day())
    assert third.message.reply_text.call_args.args[0].startswith(localized('quota', main.language))
    assert main.conversation_context[2][-1][1] == 'No search'
    assert main.db_storage.load_user_data(2)['conversation_context'][-1][1] == 'No search'

    main.grok_service.generate.return_value = GrokResult('Memory answer', notice='fallback')
    fallback = update('ботяра web: fallback', 5)
    await main.schedule_llm_message(fallback, context)
    await asyncio.gather(*app.tasks)
    assert fallback.message.reply_text.call_args.args[0].startswith(localized('fallback', main.language))
    assert main.conversation_context[5][-1][1] == 'Memory answer'

    count = main.grok_service.generate.await_count
    today = main.datetime.now().strftime('%Y-%m-%d')
    main.llm_daily_limit[11] = {'date': today, 'count': main.LLM_RPD_LIMIT}
    daily = update('ботяра daily', 11)
    await main.respond_with_llm_message(daily)
    assert daily.message.reply_text.call_args.args[0] == localized('daily_limit', main.language)
    main.llm_daily_limit[12] = {'date': today, 'count': 0}
    main.llm_rate_limit[12] = [main.time.time()] * main.LLM_RPM_LIMIT
    minute = update('ботяра minute', 12)
    await main.respond_with_llm_message(minute)
    assert minute.message.reply_text.call_args.args[0] == localized('minute_limit', main.language)
    for settings in ({'LLM_PROVIDER': 'invalid'}, {'GROK_API_KEY': ''}, {'LLM_PROVIDER': 'gemini', 'GEMINI_API_KEY': ''}):
        with patch.multiple(main, **settings):
            unavailable = update('ботяра unavailable', 13)
            await main.respond_with_llm_message(unavailable)
            assert unavailable.message.reply_text.call_args.args[0] == localized('not_configured', main.language)
    assert main.grok_service.generate.await_count == count

    main.grok_service.generate.side_effect = RuntimeError('PRIVATE_GENERATION_SENTINEL')
    failed = update('ботяра failure', 14)
    await main.respond_with_llm_message(failed)
    assert failed.message.reply_text.call_args.args[0] == localized('failure', main.language)
    main.grok_service.generate.side_effect = None

    empty = update('ботяра web:')
    count = main.grok_service.generate.await_count
    await main.handle_message(empty, context)
    assert main.grok_service.generate.await_count == count
    assert empty.message.reply_text.call_args.args[0] == localized('empty', main.language)

    main.LLM_PROVIDER = 'gemini'
    with patch.object(main, 'call_gemini_api', AsyncMock(return_value='Gemini')) as gemini:
        request = update('ботяра web: hi', 3)
        await main.schedule_llm_message(request, context)
        await asyncio.gather(*app.tasks)
        assert gemini.await_count == 1
        assert request.message.reply_text.call_args.args[0].startswith(localized('gemini', main.language))
    main.LLM_PROVIDER = 'grok'

    # Exercise the actual Gemini adapter, not a replacement of call_gemini_api.
    generator = AsyncMock(return_value=SimpleNamespace(candidates=[], text='**Readable** C# 2 * 3 https://e.test/a#part'))
    with patch.object(main.genai, 'GenerativeModel', return_value=SimpleNamespace(generate_content_async=generator)):
        result = await main.call_gemini_api(main.build_llm_messages('Question', []), 'Question')
        assert result == 'Readable C# 2 * 3 https://e.test/a#part'
        generator.side_effect = [SimpleNamespace(candidates=[SimpleNamespace(finish_reason=2)]), SimpleNamespace(text='# General information')]
        result = await main.call_gemini_api(main.build_llm_messages('Question', []), 'Question')
        assert result == localized('general_info', main.language) + '\nGeneral information'
        assert generator.await_count == 3
        assert all(call.kwargs['request_options']['retry'] is None for call in generator.call_args_list)

    with patch.object(main, 'generate_image_and_send', AsyncMock()) as image:
        await main.handle_message(update('ботяра image: Red Cat'), context)
        assert image.call_args.args[1] == 'Red Cat'
    with patch.object(main, 'respond_with_bot_message', AsyncMock()) as health:
        await main.handle_message(update('bot_health'), context)
        health.assert_awaited_once()
    with patch.object(downloads, 'is_video_too_long_to_download', return_value=False), patch.object(downloads, 'download_media', return_value=None):
        failed_download = update('https://reddit.com/r/test/comments/123')
        await main.handle_message(failed_download, context)
        assert ('Не вдалося завантажити' if main.language == 'uk' else "Couldn't download") in failed_download.message.reply_text.call_args.args[0]

    class CancelApp(App):
        def create_task(self, coroutine, **kwargs):
            task = super().create_task(coroutine, **kwargs)
            task.cancel()
            return task
    cancel_context = SimpleNamespace(application=CancelApp(), bot=context.bot)
    await main.schedule_llm_message(update('ботяра cancelled', 4), cancel_context)
    await asyncio.gather(*cancel_context.application.tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert not main.llm_gate.users and main.llm_gate.pending_count == 0
    assert not main.typing_indicators.entries
    await main.typing_indicators.close()

asyncio.run(check())
# Exercise real PTB initialize/poll/stop/shutdown against an in-memory Bot API.
import json
from telegram.request import BaseRequest
from telegram.ext import ApplicationBuilder
created = []
class OfflineRequest(BaseRequest):
    def __init__(self):
        self.stop_requested = False
    @property
    def read_timeout(self):
        return 1
    async def initialize(self):
        pass
    async def shutdown(self):
        pass
    async def do_request(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit('/', 1)[-1]
        if endpoint == 'getMe':
            result = {'id': 123456, 'is_bot': True, 'first_name': 'Test', 'username': 'unit_bot'}
        elif endpoint == 'getUpdates':
            await asyncio.sleep(.01)
            if not self.stop_requested:
                self.stop_requested = True
                created[0].stop_running()
            result = []
        else:
            result = True
        return 200, json.dumps({'ok': True, 'result': result}).encode()
builder = main.Application.builder().get_updates_request(OfflineRequest())
original_build = ApplicationBuilder.build
def build(self):
    app = original_build(self)
    created.append(app)
    return app
with patch.object(main, 'HTTPXRequest', return_value=OfflineRequest()), patch.object(main.Application, 'builder', return_value=builder), patch.object(ApplicationBuilder, 'build', build):
    main.main()
assert len(created) == 1 and not created[0].running
assert main.cleanup_task.cancelled()
print('Isolated bot integration and startup passed')
'''
        for language in ('uk', 'en'):
            with TemporaryDirectory() as temp:
                for response_language in ('uk', 'en'):
                    Path(temp, f'responses_{response_language}.json').write_bytes(
                        (root / 'src' / f'responses_{response_language}.json').read_bytes()
                    )
                env = {
                    'PATH': os.environ.get('PATH', ''),
                    'HOME': temp,
                    'PYTHONPATH': str(root / 'src'),
                    'PYTHON_DOTENV_DISABLED': '1',
                    'OPENSSL_armcap': '0',
                    'BOT_TOKEN': '123456:unit-test',
                    'GROK_API_KEY': 'unit-test',
                    'GEMINI_API_KEY': 'unit-test',
                    'USE_LLM': 'true',
                    'LIMIT_BOT_ACCESS': 'False',
                    'LANGUAGE': language,
                    'LOG_LEVEL': 'DEBUG',
                }
                result = subprocess.run(
                    [sys.executable, '-c', script],
                    cwd=temp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('Isolated bot integration and startup passed', result.stdout)
                self.assertNotIn('https://Host/AbC?Key=X', result.stderr)
                self.assertNotIn('SENTINEL', result.stdout + result.stderr)

    def test_optional_llm_config_only_blocks_startup_when_enabled(self):
        root = Path(__file__).resolve().parents[1]
        for variable in ('LLM_MAX_JOBS_PER_USER', 'GROK_SEARCH_TIMEOUT_SEC', 'LLM_RPM_LIMIT', 'MAX_CONTEXT_CHARS'):
            for enabled in ('false', 'true'):
                with self.subTest(variable=variable, enabled=enabled), TemporaryDirectory() as temp:
                    env = {
                        'PATH': os.environ.get('PATH', ''),
                        'PYTHONPATH': str(root / 'src'),
                        'PYTHON_DOTENV_DISABLED': '1',
                        'USE_LLM': enabled,
                        variable: 'invalid-number',
                    }
                    script = 'import main; assert not main.USE_LLM; main.db_storage.close()'
                    result = subprocess.run(
                        [sys.executable, '-c', script],
                        cwd=temp,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if enabled == 'false':
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('ValueError', result.stderr)
