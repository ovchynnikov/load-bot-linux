"""Behavioral coverage for search evidence, quotas, fallback and text delivery."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from db_storage import BotStorage
from grok_api import GrokService, InvalidResponse, parse_response
from llm_config import LLMSettings
from llm_delivery import send_reply
from llm_text import localized, parse_command, plain_gemini_text, render_citations, split_reply


class Response:
    def __init__(self, data):
        self.data = data

    def model_dump(self):
        return self.data


def response(state="zero", text="Answer", status="completed"):
    usage = {"num_server_side_tools_used": 0}
    tools = []
    if state == "used":
        tools = [{"type": "web_search_call", "status": "completed"}]
        usage = {"num_server_side_tools_used": 1, "server_side_tool_usage_details": {"web_search_calls": 1}}
    elif state == "unknown":
        usage = {}
    return Response(
        {
            "status": status,
            "usage": usage,
            "output": tools + [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        }
    )


class TextTests(unittest.TestCase):
    def test_leading_command_and_case(self):
        self.assertEqual(parse_command('Ботяра, WEB: https://Host/AbC?Token=X'), ('web', 'https://Host/AbC?Token=X'))
        self.assertEqual(parse_command('ботяра, web: explain "ботяра image: cat"')[0], 'web')
        self.assertEqual(parse_command('ботяра, image: Red CAT'), ('image', 'Red CAT'))
        self.assertEqual(parse_command('ботяра web: '), ('web', ''))
        self.assertEqual(parse_command('ботяра2 hi'), ('chat', '2 hi'))
        self.assertEqual(parse_command('ботярами hi'), ('chat', 'ботярами hi'))

    def test_balanced_citations_renumbering_and_context(self):
        raw = 'A[[7]](https://e.test/Topic_(detail)) B[[7]](https://e.test/b) C[[1]](https://e.test/Topic_(detail))'
        rendered, context = render_citations(raw, True, 'en')
        self.assertIn('A[1] B[2] C[1]', rendered)
        self.assertEqual(rendered.count('[1] https://e.test/Topic_(detail)'), 1)
        self.assertEqual(context, 'A B C')
        for raw in ('bad[[1]](javascript:alert(1))', 'bad[[1]](https://user:pass@e.test)', 'bad[[1]](https://e.test'):
            self.assertEqual(render_citations(raw, True)[0], raw)
        plain, _ = render_citations('A[[1]](https://e.test/a)', False)
        self.assertEqual(plain, 'Ahttps://e.test/a')
        self.assertNotIn('Джерела', plain)
        rejected, _ = render_citations('A[[1]](https://e.test/a)', True, known_sources=())
        self.assertNotIn('Джерела', rejected)
        corroborated, _ = render_citations('A[[1]](https://e.test/a)', True, known_sources=('https://e.test/a',))
        self.assertIn('Джерела', corroborated)

    def test_chunks_preserve_text_and_unicode_budget(self):
        for text in ('😀' * 4000, 'word ' * 3000, 'a' * 9000, 'abc\n\n' * 1200):
            chunks = split_reply(text)
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(chunk.encode('utf-16-le')) // 2 <= 3800 for chunk in chunks))
        self.assertNotEqual(localized('fallback', 'uk'), localized('fallback', 'en'))

    def test_config_rejects_invalid_budgets(self):
        for changes in (
            {'search_timeout': 121},
            {'timeout': float('nan')},
            {'global_quota': -1},
            {'concurrency': 0},
            {'jobs_per_user': 0},
            {'search_mode': 'always'},
            {'reasoning': 'banana'},
        ):
            with self.assertRaises(ValueError):
                replace(LLMSettings(), **changes)

    def test_live_evidence_and_missing_usage(self):
        artifact = json.loads(
            (Path(__file__).resolve().parent / 'fixtures/grok-compatibility-results.json').read_text()
        )
        for probe in [artifact['protocol_probe']] + artifact['sdk_probes']:
            if probe.get('status') != 'completed':
                continue
            raw = response('zero' if not probe['tool_outputs'] else 'used').data
            raw['usage'] = probe['usage']
            self.assertEqual(
                parse_response(Response(raw)).search_state, 'zero' if not probe['tool_outputs'] else 'used'
            )
        self.assertEqual(parse_response(response('unknown')).search_state, 'unknown')
        raw = response('used').data
        raw['usage']['num_server_side_tools_used'] = 0
        parsed = parse_response(Response(raw))
        self.assertEqual(parsed.search_state, 'used')
        self.assertFalse(parsed.refund_eligible)
        raw = response('zero').data
        raw['usage'] = {'num_server_side_tools_used': True, 'server_side_tool_usage_details': {'web_search_calls': 0}}
        self.assertEqual(parse_response(Response(raw)).search_state, 'unknown')
        for status, text in [('incomplete', 'partial'), ('completed', '')]:
            with self.assertRaises(InvalidResponse):
                parse_response(response(text=text, status=status))

    def test_gemini_cleanup_preserves_literal_content(self):
        raw = '# Heading\n**bold** and *italic*, `code`.\n```python\nx = 2 * 3\n```\nC#; 2 ** 3; https://e.test/a#part'
        plain = plain_gemini_text(raw)
        self.assertIn('Heading\nbold and italic, code.', plain)
        self.assertIn('x = 2 * 3', plain)
        self.assertIn('C#; 2 ** 3; https://e.test/a#part', plain)
        self.assertNotIn('```', plain)
        literal = '2**3 + 4**5; a*b*c; https://e.test/**path**#part'
        self.assertEqual(plain_gemini_text(literal), literal)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = str(Path(self.temp.name) / 'test.db')
        self.db = BotStorage(self.path)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def test_concurrent_atomic_limits_restart_refund_and_midnight(self):
        def reserve(user):
            return self.db.reserve_search(user, 2, 5, '2026-10-17')

        with ThreadPoolExecutor(max_workers=8) as pool:
            reservations = list(pool.map(reserve, range(20)))
        accepted = [item for item in reservations if item]
        self.assertEqual(len(accepted), 5)
        self.db.close()
        self.db = BotStorage(self.path)
        self.assertIsNone(self.db.reserve_search(99, 2, 5, '2026-10-17'))
        self.assertIsNotNone(self.db.reserve_search(99, 2, 5, '2026-10-18'))
        self.db.settle_search(accepted[0], True)
        self.db.settle_search(accepted[0], True)
        self.assertIsNotNone(self.db.reserve_search(99, 2, 5, '2026-10-17'))
        self.assertIsNone(self.db.reserve_search(99, 2, 5, '2026-10-17'))

    def test_user_limit_failure_has_no_global_side_effect_and_zero_disables(self):
        self.assertIsNotNone(self.db.reserve_search(1, 1, 2))
        self.assertIsNone(self.db.reserve_search(1, 1, 2))
        self.assertIsNotNone(self.db.reserve_search(2, 1, 2))
        self.assertIsNone(self.db.reserve_search(3, 0, 2))

    def test_field_safe_saves_and_notice_survive_user_cleanup(self):
        self.db.update_user_image_limits(1, [1.0], 7, '2026-10-17')
        self.db.save_user_data(1, [['q', 'a']], [2.0], 3, '2026-10-17', 5.0)
        self.assertEqual(self.db.load_user_data(1)['img_gen_daily_count'], 7)
        self.db.update_user_image_limits(1, [2.0], 8, '2026-10-17')
        self.assertEqual(self.db.load_user_data(1)['daily_count'], 3)
        self.db.mark_search_notice(1, '2026-10-17')
        self.db.delete_user_data(1)
        self.assertTrue(self.db.search_notice_shown(1, '2026-10-17'))


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.db = BotStorage(str(Path(self.temp.name) / 'test.db'))
        self.create = AsyncMock()
        self.client = SimpleNamespace(responses=SimpleNamespace(create=self.create))
        self.settings = LLMSettings(search_enabled=True, search_timeout=0.02, fallback_timeout=0.05, timeout=0.15)
        self.service = GrokService(self.client, self.db, self.settings)
        self.messages = [{'role': 'user', 'content': 'Question'}]

    async def asyncTearDown(self):
        self.db.close()
        self.temp.cleanup()

    async def test_success_request_contract_and_zero_refund(self):
        self.create.return_value = response('used')
        result = await self.service.generate(self.messages, 1, explicit=True)
        self.assertEqual(result.search_state, 'used')
        args = self.create.call_args.kwargs
        self.assertEqual(args['tool_choice'], 'required')
        self.assertEqual(args['extra_body'], {'max_turns': 3})
        self.assertFalse(args['store'])
        self.create.return_value = response('zero')
        await self.service.generate(self.messages, 2)
        self.assertIsNotNone(self.db.reserve_search(2, 1, 50))

    async def test_timeout_leaves_reserved_fallback_time_and_retains_quota(self):
        async def answer(**kwargs):
            if kwargs.get('tools'):
                await asyncio.sleep(1)
            return response()

        self.create.side_effect = answer
        result = await self.service.generate(self.messages, 1, explicit=True)
        self.assertEqual(result.notice, 'fallback')
        self.assertEqual(self.create.await_count, 2)
        self.assertNotIn('tools', self.create.call_args.kwargs)
        self.assertIsNone(self.db.reserve_search(1, 1, 50))

    async def test_daily_notice_and_explicit_warning_policy(self):
        self.service.settings = replace(self.settings, global_quota=0)
        self.create.return_value = response()
        first = await self.service.generate(self.messages, 1)
        self.assertEqual(first.notice, 'quota')
        self.db.mark_search_notice(1, first.notice_day)
        second = await self.service.generate(self.messages, 1)
        self.assertIsNone(second.notice)
        for _ in range(2):
            explicit = await self.service.generate(self.messages, 1, True)
            self.assertEqual(explicit.notice, 'fallback')
        self.assertTrue(all('tools' not in call.kwargs for call in self.create.call_args_list))

    async def test_incomplete_unknown_and_failed_search_get_one_fallback(self):
        for bad in (response(status='incomplete'), response('unknown'), RuntimeError('error')):
            self.create.reset_mock()
            self.create.side_effect = [bad, response()]
            result = await self.service.generate(self.messages, 1, True)
            self.assertEqual(result.notice, 'fallback')
            self.assertEqual(self.create.await_count, 2)
        self.create.side_effect = [RuntimeError('search'), RuntimeError('fallback')]
        with self.assertRaises(RuntimeError):
            await self.service.generate(self.messages, 1, True)
        self.assertEqual(self.create.await_count, 4)

    async def test_disabled_search_and_explicit_only(self):
        self.create.return_value = response()
        self.service.settings = replace(self.settings, search_enabled=False)
        result = await self.service.generate(self.messages, 1, True)
        self.assertEqual(result.notice, 'fallback')
        self.assertNotIn('tools', self.create.call_args.kwargs)
        self.service.settings = replace(self.settings, search_mode='explicit')
        await self.service.generate(self.messages, 1)
        self.assertNotIn('tools', self.create.call_args.kwargs)

    async def test_execution_and_refund_matrix_in_both_modes(self):
        # Each row is (tool statuses, usage, expected search state, refund).
        cases = [
            (['completed'], None, 'used', False),
            (['completed'], {'num_server_side_tools_used': 0}, 'used', False),
            (['completed'], {'num_server_side_tools_used': 'bad'}, 'used', False),
            (
                ['completed'],
                {'num_server_side_tools_used': 0, 'server_side_tool_usage_details': {'web_search_calls': 1}},
                'used',
                False,
            ),
            (['completed', 'failed'], {}, 'used', False),
            ([], {'num_server_side_tools_used': 0}, 'zero', True),
            ([], {'server_side_tool_usage_details': {'web_search_calls': 0}}, 'zero', True),
            ([], None, 'unknown', False),
            ([], {'num_server_side_tools_used': 2}, 'unknown', False),
            ([], {'num_server_side_tools_used': True}, 'unknown', False),
            (
                [],
                {'num_server_side_tools_used': 0, 'server_side_tool_usage_details': {'web_search_calls': 1}},
                'unknown',
                False,
            ),
            ([], 'malformed', 'unknown', False),
            (['failed'], {'num_server_side_tools_used': 0}, 'failed', False),
            (['in_progress'], None, 'failed', False),
        ]
        for explicit in (False, True):
            for index, (statuses, usage, state, refund) in enumerate(cases):
                with self.subTest(explicit=explicit, statuses=statuses, usage=usage):
                    raw = response('unknown', text='Original answer').data
                    raw['output'][:0] = [{'type': 'web_search_call', 'status': status} for status in statuses]
                    if usage is None:
                        raw.pop('usage')
                    else:
                        raw['usage'] = usage
                    parsed = parse_response(Response(raw))
                    self.assertEqual((parsed.search_state, parsed.refund_eligible), (state, refund))
                    self.create.reset_mock()
                    self.create.side_effect = [Response(raw), response(text='Fallback answer')]
                    user = 100 + index + int(explicit) * 100
                    result = await self.service.generate(self.messages, user, explicit)
                    fallback = state == 'failed' or (explicit and state != 'used')
                    self.assertEqual(self.create.await_count, 2 if fallback else 1)
                    self.assertEqual(result.text, 'Fallback answer' if fallback else 'Original answer')
                    self.assertEqual(result.notice, 'fallback' if fallback else None)
                    self.assertEqual(self.db.reserve_search(user, 1, 1000) is not None, refund)

    async def test_auto_nonterminal_empty_and_provider_failures_keep_labeled_fallback(self):
        for index, bad in enumerate(
            (response(status='incomplete'), response(status='failed'), response(text=''), RuntimeError('private error'))
        ):
            with self.subTest(index=index):
                self.create.reset_mock()
                self.create.side_effect = [bad, response(text='Fallback')]
                result = await self.service.generate(self.messages, 300 + index)
                self.assertEqual((result.text, result.notice), ('Fallback', 'fallback'))
                self.assertEqual(self.create.await_count, 2)
                self.assertIsNone(self.db.reserve_search(300 + index, 1, 1000))

    async def test_tool_free_instruction_is_leading_and_does_not_mutate_history(self):
        self.service.settings = replace(self.settings, search_enabled=False)
        self.create.return_value = response()
        for messages in (self.messages, [{'role': 'system', 'content': 'Persona'}] + self.messages):
            original = [dict(item) for item in messages]
            await self.service.generate(messages, 1)
            sent = self.create.call_args.kwargs['input']
            self.assertEqual(sent[0]['role'], 'system')
            self.assertIn('Do not claim online verification', sent[0]['content'])
            self.assertEqual(sent[-1], self.messages[-1])
            self.assertEqual(messages, original)
            if messages[0]['role'] == 'system':
                self.assertTrue(sent[0]['content'].startswith('Persona\n'))

    async def test_cancelled_dispatch_retains_reservation_without_fallback(self):
        started = asyncio.Event()

        async def wait(**_kwargs):
            started.set()
            await asyncio.sleep(10)

        self.create.side_effect = wait
        job = asyncio.create_task(self.service.generate(self.messages, 1, True))
        await started.wait()
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertEqual(self.create.await_count, 1)
        self.assertIsNone(self.db.reserve_search(1, 1, 50))

    async def test_partial_delivery_threads_every_chunk_and_never_retries(self):
        message = SimpleNamespace(
            message_id=7,
            message_thread_id=9,
            is_topic_message=False,
            reply_text=AsyncMock(side_effect=[None, RuntimeError('delivery')]),
        )
        delivered = AsyncMock()
        with self.assertRaises(RuntimeError):
            await send_reply(SimpleNamespace(message=message), '😀' * 4000, delivered)
        self.assertEqual(delivered.await_count, 1)
        self.assertEqual(message.reply_text.await_count, 2)
        for call in message.reply_text.call_args_list:
            self.assertEqual(call.kwargs['reply_to_message_id'], 7)
            self.assertIsNone(call.kwargs['message_thread_id'])
            self.assertIsNone(call.kwargs['parse_mode'])


class SDKContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_serialization_and_xai_extensions_without_network(self):
        import httpx
        from openai import AsyncOpenAI

        received = []

        async def handler(request):
            body = json.loads(request.content)
            received.append(body)
            raw = response('used').data
            raw.update(id='unit-response', object='response', created_at=0, model='grok-4.3')
            raw['usage']['cost_in_usd_ticks'] = 123
            return httpx.Response(200, json=raw)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with AsyncOpenAI(
                api_key='unit-test', base_url='https://api.x.ai/v1', http_client=http, max_retries=0
            ) as client:
                with TemporaryDirectory() as temp:
                    db = BotStorage(str(Path(temp) / 'sdk.db'))
                    try:
                        result = await GrokService(client, db, LLMSettings(search_enabled=True)).generate(
                            [{'role': 'user', 'content': 'test'}], 1, True
                        )
                    finally:
                        db.close()
        self.assertEqual(result.search_state, 'used')
        self.assertEqual(result.usage['cost_in_usd_ticks'], 123)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['max_turns'], 3)
        self.assertEqual(received[0]['tool_choice'], 'required')
        self.assertEqual(received[0]['max_output_tokens'], 4096)
        self.assertFalse(received[0]['store'])


class AdditionalStorageTests(unittest.TestCase):
    def test_global_limit_across_independent_connections(self):
        with TemporaryDirectory() as temp:
            path = str(Path(temp) / 'shared.db')
            databases = [BotStorage(path) for _ in range(4)]
            try:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    accepted = list(pool.map(lambda i: databases[i % 4].reserve_search(i, 1, 3), range(20)))
                self.assertEqual(sum(item is not None for item in accepted), 3)
            finally:
                for db in databases:
                    db.close()

    def test_partial_refund_failure_rolls_back_both_scopes(self):
        with TemporaryDirectory() as temp:
            db = BotStorage(str(Path(temp) / 'rollback.db'))
            try:
                reservation = db.reserve_search(1, 1, 1)
                db.conn.execute(
                    "CREATE TRIGGER fail_refund BEFORE UPDATE ON search_quota WHEN NEW.scope='global' BEGIN SELECT RAISE(ABORT, 'test failure'); END"
                )
                with self.assertRaises(Exception):
                    db.settle_search(reservation, True)
                self.assertEqual([row[0] for row in db.conn.execute('SELECT count FROM search_quota')], [1, 1])
                db.conn.execute('DROP TRIGGER fail_refund')
                db.settle_search(reservation, True)
                self.assertIsNotNone(db.reserve_search(2, 1, 1))
            finally:
                db.close()
