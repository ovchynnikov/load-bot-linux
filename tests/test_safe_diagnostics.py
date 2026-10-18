"""Error logs/admin alerts retain useful evidence without arbitrary input bodies."""

import os
from datetime import timedelta
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import BadRequestError, AuthenticationError
from telegram import Update
from telegram.error import BadRequest as TelegramBadRequest, ChatMigrated, Forbidden, NetworkError, RetryAfter

import general_error_handler
from safe_diagnostics import download_diagnostic_url, exception_summary


class DiagnosticTests(unittest.TestCase):
    def test_telegram_reasons_are_actionable_without_arbitrary_descriptions(self):
        for exception, reason in (
            (TelegramBadRequest('Message thread not found'), 'message_thread_not_found'),
            (TelegramBadRequest('Bad Request: replied message not found'), 'replied_message_not_found'),
            (TelegramBadRequest('not enough rights to send text messages to the chat'), 'insufficient_send_rights'),
            (Forbidden('Forbidden: bot was blocked by the user'), 'bot_blocked_by_user'),
        ):
            self.assertIn(f'telegram_error={reason}', exception_summary(exception))
        for exception in (
            TelegramBadRequest('PRIVATE_SENTINEL'),
            TelegramBadRequest('Message thread not found PRIVATE_SENTINEL'),
            Forbidden('Forbidden: bot was blocked by the user PRIVATE_SENTINEL'),
            NetworkError('Transport failed for https://api.telegram.org/botPRIVATE_SENTINEL/sendMessage'),
        ):
            summary = exception_summary(exception)
            self.assertNotIn('SENTINEL', summary)
            self.assertNotIn('telegram_error=', summary)

    def test_telegram_retry_migration_and_nested_cause_metadata(self):
        for use_timedelta in ('0', '1'):
            with patch.dict(os.environ, {'PTB_TIMEDELTA': use_timedelta}):
                summary = exception_summary(RetryAfter(timedelta(seconds=12)))
                self.assertIn('telegram_error=flood_control', summary)
                self.assertIn('retry_after_seconds=12', summary)
        summary = exception_summary(ChatMigrated(-100123))
        self.assertIn('telegram_error=chat_migrated', summary)
        self.assertIn('new_chat_id=-100123', summary)
        with self.assertRaises(RuntimeError) as caught:
            try:
                raise TelegramBadRequest('Message thread not found')
            except TelegramBadRequest as cause:
                raise RuntimeError('PRIVATE_SENTINEL') from cause
        summary = exception_summary(caught.exception)
        self.assertIn('telegram_error=message_thread_not_found', summary)
        self.assertNotIn('SENTINEL', summary)

    def test_provider_status_code_request_id_and_chains_without_sensitive_text(self):
        for exception_class, status, code, reason in (
            (BadRequestError, 400, 'model_not_found', 'invalid_request'),
            (AuthenticationError, 401, 'invalid_api_key', 'authentication_failed'),
        ):
            request = httpx.Request(
                'POST',
                'https://api.x.ai/v1/responses?secret=PRIVATE_SENTINEL',
                headers={'Authorization': 'PRIVATE_SENTINEL'},
            )
            response = httpx.Response(status, request=request, headers={'x-request-id': 'req_1234567890abcdef'})
            exception = exception_class(
                'PRIVATE_SENTINEL', response=response, body={'code': code, 'message': 'PRIVATE_SENTINEL'}
            )
            with self.assertRaises(exception_class) as caught:
                try:
                    raise RuntimeError('CHAIN_SENTINEL')
                except RuntimeError as cause:
                    raise exception from cause
            diagnostic = exception_summary(caught.exception)
            self.assertIn(f'status={status}', diagnostic)
            self.assertIn(f'code={code}', diagnostic)
            self.assertIn(f'reason={reason}', diagnostic)
            self.assertIn('request_id=req_1234567890abcdef', diagnostic)
            self.assertIn('test_safe_diagnostics.py:', diagnostic)
            self.assertIn('type=RuntimeError', diagnostic)
            self.assertNotIn('SENTINEL', diagnostic)
            self.assertNotIn('Authorization', diagnostic)

    def test_timeout_subprocess_and_unknown_codes_are_safe_and_distinct(self):
        self.assertIn('reason=timeout', exception_summary(TimeoutError('PRIVATE_SENTINEL')))
        diagnostic = exception_summary(
            subprocess.CalledProcessError(7, ['ffmpeg', 'PRIVATE_SENTINEL'], stderr='STDERR_SENTINEL')
        )
        self.assertIn('exit_code=7', diagnostic)
        self.assertNotIn('SENTINEL', diagnostic)
        exception = RuntimeError('PRIVATE_SENTINEL')
        exception.code = 'CODE_SENTINEL'
        exception.request_id = 'REQUEST_SENTINEL'
        self.assertNotIn('SENTINEL', exception_summary(exception))

    def test_only_standalone_supported_download_url_is_included(self):
        self.assertEqual(
            download_diagnostic_url('https://USER:PASS@www.reddit.com/r/test?KEY=SECRET#FRAGMENT'),
            'https://www.reddit.com/r/test',
        )
        for text in (
            'ботяра web: https://reddit.com/r/test',
            'https://reddit.com/r/test extra PRIVATE_SENTINEL',
            'https://reddit.com.evil.test/a',
            'https://private.test/a',
            None,
        ):
            self.assertIsNone(download_diagnostic_url(text))


class AdminDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_reason_reaches_logs_and_admin_alerts(self):
        context = SimpleNamespace(
            error=TelegramBadRequest('Message thread not found'), bot=SimpleNamespace(send_message=AsyncMock())
        )
        with patch.object(general_error_handler, 'send_error_to_admin', True), patch.object(
            general_error_handler, 'admins_chat_ids', '1'
        ), self.assertLogs('logger', level='ERROR') as logs:
            await general_error_handler.error_handler(None, context)
        self.assertIn('telegram_error=message_thread_not_found', logs.output[0])
        self.assertIn('telegram_error=message_thread_not_found', context.bot.send_message.call_args.kwargs['text'])

    async def test_logs_and_admin_alerts_are_safe_with_and_without_a_user(self):
        for language in ('uk', 'en'):
            for text in ('ботяра PRIVATE_SENTINEL', 'https://user:pass@reddit.com/r/test?KEY=PRIVATE_SENTINEL#SECRET'):
                update = Update.de_json(
                    {
                        'update_id': 1,
                        'message': {
                            'message_id': 2,
                            'date': 0,
                            'chat': {'id': 1, 'type': 'private'},
                            'from': {'id': 1, 'first_name': 'Name', 'is_bot': False},
                            'text': text,
                        },
                    },
                    None,
                )
                context = SimpleNamespace(
                    error=RuntimeError('ERROR_SENTINEL'), bot=SimpleNamespace(send_message=AsyncMock())
                )
                with patch.object(general_error_handler, 'send_error_to_admin', True), patch.object(
                    general_error_handler, 'admins_chat_ids', '1,2'
                ), patch.dict(os.environ, {'LANGUAGE': language}), self.assertLogs('logger', level='ERROR') as logs:
                    await general_error_handler.error_handler(update, context)
                    await general_error_handler.error_handler(None, context)
                self.assertEqual(context.bot.send_message.await_count, 4)
                combined = '\n'.join(logs.output) + '\n'.join(
                    call.kwargs['text'] for call in context.bot.send_message.call_args_list
                )
                self.assertNotIn('SENTINEL', combined)
                self.assertNotIn('user:pass', combined)
                self.assertIn('type=RuntimeError', combined)
                if text.startswith('https:'):
                    self.assertIn('https://reddit.com/r/test', combined)
                else:
                    self.assertNotIn('ботяра', combined)
