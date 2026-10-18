"""Check actual PTB Bot API payloads rather than forwarding attributes on fakes."""

import asyncio
import json
import unittest

from telegram import Bot, Message, Update
from telegram.request import BaseRequest

from llm_delivery import send_reply
from llm_jobs import TypingIndicators
from llm_text import localized


class RecordingRequest(BaseRequest):
    def __init__(self):
        self.calls = []
        self.action_sent = asyncio.Event()

    @property
    def read_timeout(self):
        return 1

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(
        self,
        url,
        method,
        request_data=None,
        read_timeout=None,
        write_timeout=None,
        connect_timeout=None,
        pool_timeout=None,
    ):
        endpoint = url.rsplit('/', 1)[-1]
        params = request_data.parameters if request_data else {}
        self.calls.append((endpoint, params))
        if endpoint == 'getMe':
            result = {'id': 123456, 'is_bot': True, 'first_name': 'Unit', 'username': 'unit_bot'}
        elif endpoint == 'sendMessage':
            result = {
                'message_id': 99,
                'date': 0,
                'chat': {'id': params['chat_id'], 'type': 'supergroup'},
                'text': params['text'],
            }
        else:
            result = True
            self.action_sent.set()
        return 200, json.dumps({'ok': True, 'result': result}).encode()


class TelegramContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_and_typing_topic_business_and_chunk_contract(self):
        request = RecordingRequest()
        async with Bot('123456:unit-test', request=request, get_updates_request=RecordingRequest()) as bot:
            for chat_type, topic, incoming_thread, business in (
                ('private', False, None, None),
                ('group', False, None, None),
                ('supergroup', False, 123, None),
                ('supergroup', True, 42, None),
                ('private', False, None, 'business-unit'),
            ):
                with self.subTest(chat=chat_type, topic=topic, business=business):
                    request.calls.clear()
                    request.action_sent.clear()
                    message = Message.de_json(
                        {
                            'message_id': 17,
                            'date': 0,
                            'chat': {
                                'id': -100999 if chat_type != 'private' else 99,
                                'type': chat_type,
                                'is_forum': topic,
                            },
                            'text': 'ботяра hi',
                            'is_topic_message': topic,
                            'message_thread_id': incoming_thread,
                            'business_connection_id': business,
                        },
                        bot,
                    )
                    update = Update(update_id=1, message=message)
                    indicators = TypingIndicators()
                    lease = indicators.acquire(bot, message)
                    try:
                        await asyncio.wait_for(request.action_sent.wait(), 1)
                        for text in ('Answer', localized('busy', 'uk'), localized('failure', 'en'), '😀' * 4000):
                            await send_reply(update, text)
                        expected_topic = incoming_thread if topic else None
                        self.assertEqual(lease.key, (message.chat_id, expected_topic, business))
                        sends = []
                        for endpoint, payload in request.calls:
                            self.assertEqual(payload.get('message_thread_id'), expected_topic)
                            self.assertEqual(payload.get('business_connection_id'), business)
                            if endpoint == 'sendMessage':
                                sends.append(payload)
                                self.assertEqual(payload['reply_parameters']['message_id'], 17)
                                self.assertNotIn('parse_mode', payload)
                        self.assertEqual(len(sends), 6)
                        self.assertEqual(''.join(item['text'] for item in sends[3:]), '😀' * 4000)
                    finally:
                        await indicators.release(lease)
                        await indicators.close()

    async def test_non_topic_reply_chains_share_the_same_chat_indicator(self):
        request = RecordingRequest()
        async with Bot('123456:unit-test', request=request, get_updates_request=RecordingRequest()) as bot:
            indicators = TypingIndicators()
            leases = []
            try:
                for thread in (123, 456):
                    message = Message.de_json(
                        {
                            'message_id': thread + 1,
                            'date': 0,
                            'chat': {'id': -100999, 'type': 'supergroup'},
                            'text': 'test',
                            'message_thread_id': thread,
                            'is_topic_message': False,
                        },
                        bot,
                    )
                    leases.append(indicators.acquire(bot, message))
                self.assertEqual(len(indicators.entries), 1)
                await indicators.release(leases[0])
                self.assertEqual(len(indicators.entries), 1)
            finally:
                for lease in leases:
                    await indicators.release(lease)
                await indicators.close()
