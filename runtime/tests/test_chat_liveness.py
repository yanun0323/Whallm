from __future__ import annotations

import http.client
import json
import socket
import struct
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runtime.tests.test_server import FakeRuntime
from deepseek_v4_ssd.cancellation import _cancellation, check_cancelled
from deepseek_v4_ssd.generation import GeneratedPiece
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, ModelManager, ModelSpec
from deepseek_v4_ssd.server import APIError, OpenAIServer
from deepseek_v4_ssd.tool_codec import QwenToolCodec, QwenToolStreamParser


class ChatLivenessTests(unittest.TestCase):
    model_id = 'swift1.5-qwen3.8-flash-next-mxfp4'

    def setUp(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.closed = threading.Event()
        self.owner_threads = []
        self.raw = 'Hello'
        self.runtime = FakeRuntime()
        self.runtime.installed = SimpleNamespace(
            root='/tmp/model', model_kind='swift1.5-qwen3.8-flash-next', has_dspark=False,
        )
        self.runtime.parse_chat = QwenToolCodec(None).parse
        self.runtime.make_tool_stream_parser = QwenToolStreamParser

        def stream(prompt, options):
            self.owner_threads.append(threading.get_ident())
            self.entered.set()
            try:
                if not self.release.wait(3):
                    raise RuntimeError('test generation gate timed out')
                yield GeneratedPiece(self.raw, 1, 5, 1, 'stop')
            finally:
                self.owner_threads.append(threading.get_ident())
                self.closed.set()

        self.runtime.stream = stream
        self.manager = ModelManager([
            ModelSpec(self.model_id, None, '/tmp/model', 'swift1.5-qwen3.8-flash-next',
                      RuntimeConfig(), ModelDefaults(64, 0, 1, 0)),
        ], runtime_loader=lambda _: self.runtime, clear_cache=lambda: None)
        self.interval = patch('deepseek_v4_ssd.server.CHAT_HEARTBEAT_SECONDS', 0.02, create=True)
        self.interval.start()
        self.server = OpenAIServer(('127.0.0.1', 0), self.manager, log_level='error')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=0.5)

    def tearDown(self):
        self.release.set()
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.manager.close()
        self.interval.stop()

    def start(self, *, tools=False, thinking=False, include_usage=True):
        payload = {
            'model': self.model_id, 'messages': [{'role': 'user', 'content': 'Hello'}],
            'stream': True, 'max_tokens': 64,
            'thinking_mode': 'thinking' if thinking else 'chat',
            'stream_options': {'include_usage': include_usage},
        }
        if tools:
            payload['tools'] = [{'type': 'function', 'function': {
                'name': 'bash', 'parameters': {'type': 'object', 'properties': {
                    'command': {'type': 'string'},
                }, 'required': ['command']},
            }}]
        self.connection.request('POST', '/v1/chat/completions', json.dumps(payload),
                                {'Content-Type': 'application/json'})
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        first = self.event(response)
        self.assertEqual(first['choices'][0]['delta'], {'role': 'assistant'})
        return response, first

    def event(self, response):
        while line := response.readline():
            if line.startswith(b'data: '):
                self.assertNotEqual(line, b'data: [DONE]\n', 'stream finished before expected event')
                result = json.loads(line[6:])
                self.assertEqual(response.readline(), b'\n')
                return result
        self.fail('Stream closed before the next event')

    def assert_heartbeat(self, event, first):
        for field in ('id', 'object', 'model', 'created', 'approximation'):
            self.assertEqual(event[field], first[field])
        self.assertEqual(event['choices'], [{'index': 0, 'delta': {}, 'finish_reason': None}])
        self.assertNotIn('usage', event)

    def finish(self, response):
        self.release.set()
        body = response.read()
        self.assertTrue(body.endswith(b'data: [DONE]\n\n'), body)
        self.assertEqual(body.count(b'data: [DONE]'), 1)
        self.assertNotIn(b'HTTP/1.1', body)
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b'data: {')]
        return events

    def assert_model_released(self):
        self.assertTrue(self.closed.wait(1))
        self.assertTrue(self.manager._generation_lock.acquire(timeout=1))
        self.manager._generation_lock.release()

    def test_prefill_wait_sends_empty_deltas_without_changing_text_or_usage(self):
        response, first = self.start()
        self.assertTrue(self.entered.wait(1))
        self.assert_heartbeat(self.event(response), first)
        self.assert_heartbeat(self.event(response), first)
        self.assertEqual(self.server.metrics.snapshot()['generation_tokens'], 0)
        events = self.finish(response)
        deltas = [choice['delta'] for event in events for choice in event.get('choices', [])]
        self.assertEqual(''.join(delta.get('content', '') for delta in deltas), 'Hello')
        self.assertEqual(events[-2]['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(events[-1]['usage'], {'prompt_tokens': 5, 'completion_tokens': 1, 'total_tokens': 6})
        self.assert_model_released()
        self.assertEqual(len(set(self.owner_threads)), 1, 'generation and cleanup changed threads')
        self.assertNotEqual(self.owner_threads[0], threading.get_ident())

    def test_prefill_longer_than_client_idle_timeout_still_completes(self):
        self.connection.timeout = 0.15
        def stream(prompt, options):
            time.sleep(0.4)
            yield GeneratedPiece('Hello', 1, 5, 1, 'stop')
        self.runtime.stream = stream
        started = time.monotonic()
        response, first = self.start()
        events = self.finish(response)
        self.assertGreaterEqual(time.monotonic() - started, 0.4)
        heartbeats = [event for event in events
                      if event.get('choices') == [{'index': 0, 'delta': {}, 'finish_reason': None}]]
        self.assertGreaterEqual(len(heartbeats), 2)
        for event in heartbeats:
            self.assert_heartbeat(event, first)
        self.assertEqual(events[-1]['usage']['completion_tokens'], 1)

    def test_tool_prefill_wait_preserves_reasoning_calls_and_usage(self):
        self.raw = ('plan</think><tool_call><function=bash><parameter=command>pwd'
                    '</parameter></function></tool_call>\n\n'
                    '<tool_call><function=bash><parameter=command>ls'
                    '</parameter></function></tool_call>')
        response, first = self.start(tools=True, thinking=True)
        self.assert_heartbeat(self.event(response), first)
        self.assert_heartbeat(self.event(response), first)
        events = self.finish(response)
        deltas = [choice['delta'] for event in events for choice in event.get('choices', [])]
        self.assertEqual(''.join(delta.get('reasoning_content', '') for delta in deltas), 'plan')
        calls = [call for delta in deltas for call in delta.get('tool_calls', [])]
        starts = [call for call in calls if 'id' in call]
        self.assertEqual([call['index'] for call in starts], [0, 1])
        self.assertEqual(len({call['id'] for call in starts}), 2)
        for index, command in enumerate(('pwd', 'ls')):
            arguments = ''.join(call['function'].get('arguments', '') for call in calls if call['index'] == index)
            self.assertEqual(json.loads(arguments), {'command': command})
        self.assertEqual(events[-2]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(events[-1]['usage']['completion_tokens'], 1)
        self.assert_model_released()
        self.assertEqual(len(set(self.owner_threads)), 1)

    def test_paused_partial_tool_stays_live_without_exposing_invalid_call(self):
        def stream(prompt, options):
            yield GeneratedPiece('<tool_call><function=bash>', 1, 5, 1, None)
            self.entered.set()
            self.release.wait(3)
            yield GeneratedPiece('<parameter=command>pwd</parameter></function></tool_call>', 2, 5, 2, 'stop')
        self.runtime.stream = stream
        response, first = self.start(tools=True, include_usage=False)
        self.assertTrue(self.entered.wait(1))
        self.assert_heartbeat(self.event(response), first)
        events = self.finish(response)
        self.assertTrue(any(choice['delta'].get('tool_calls') for event in events for choice in event.get('choices', [])))
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertFalse(any('usage' in event for event in events))

    def test_generation_errors_end_with_error_and_done_not_bare_eof(self):
        for tools in (False, True):
            for error in (RuntimeError('private failure detail'),
                          APIError('Test limit reached.', code='test_limit')):
                with self.subTest(tools=tools, error=type(error).__name__):
                    self.release.clear()
                    def broken(prompt, options):
                        if not self.release.wait(3):
                            raise RuntimeError('test generation gate timed out')
                        raise error
                        yield
                    self.runtime.stream = broken
                    response, first = self.start(tools=tools)
                    self.assert_heartbeat(self.event(response), first)
                    events = self.finish(response)
                    self.assertIn('error', events[-1])
                    expected = 'test_limit' if isinstance(error, APIError) else 'server_error'
                    self.assertEqual(events[-1]['error']['code'], expected)
                    self.assertNotIn('private failure detail', json.dumps(events))
                    self.assertFalse(any(choice.get('finish_reason') for event in events for choice in event.get('choices', [])))
                    self.assertTrue(self.manager._generation_lock.acquire(timeout=1))
                    self.manager._generation_lock.release()

    def test_error_after_partial_output_is_not_reported_as_success(self):
        def stream(prompt, options):
            yield GeneratedPiece('Partial answer', 1, 5, 1, None)
            raise RuntimeError('private mid-stream failure')
        self.runtime.stream = stream
        response, _ = self.start()
        events = self.finish(response)
        self.assertEqual(events[0]['choices'][0]['delta'], {'content': 'Partial answer'})
        self.assertEqual(events[-1]['error']['code'], 'server_error')
        self.assertFalse(any('usage' in event for event in events))
        self.assertFalse(any(choice.get('finish_reason') for event in events for choice in event.get('choices', [])))

    def test_invalid_tool_output_has_no_success_or_usage_after_error(self):
        self.raw = '<tool_call><function=bash>'
        response, first = self.start(tools=True)
        self.assert_heartbeat(self.event(response), first)
        events = self.finish(response)
        self.assertEqual(events[-1]['error']['code'], 'invalid_tool_call')
        self.assertFalse(any('usage' in event for event in events))
        self.assertFalse(any(choice.get('finish_reason') for event in events for choice in event.get('choices', [])))

    def test_disconnect_during_prefill_cancels_and_next_request_recovers(self):
        for tools, reset in ((False, False), (True, False), (True, True)):
            with self.subTest(tools=tools, reset=reset):
                self.closed.clear()
                self.entered.clear()
                owner_threads = []
                def stream(prompt, options):
                    owner_threads.append(threading.get_ident())
                    self.entered.set()
                    try:
                        if not _cancellation.get().wait(2):
                            raise RuntimeError('disconnect not observed')
                        check_cancelled()
                        yield GeneratedPiece('unused', 1, 5, 1, 'stop')
                    finally:
                        owner_threads.append(threading.get_ident())
                        self.closed.set()
                self.runtime.stream = stream
                response, _ = self.start(tools=tools)
                self.assertTrue(self.entered.wait(1))
                if reset:
                    response.fp.raw._sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                response.close()
                self.connection.close()
                self.assert_model_released()
                self.assertEqual(len(set(owner_threads)), 1)
                self.runtime.stream = lambda prompt, options: iter([GeneratedPiece('Recovered', 1, 5, 1, 'stop')])
                response, _ = self.start(tools=tools)
                self.assertIn('Recovered', json.dumps(self.finish(response)))

    def test_heartbeat_write_failure_closes_generator_on_its_owner_thread(self):
        from deepseek_v4_ssd.server import OpenAIHandler
        failed = threading.Event()
        original = OpenAIHandler._sse
        def fail_heartbeat(handler, value):
            if value.get('choices') == [{'index': 0, 'delta': {}, 'finish_reason': None}]:
                failed.set()
                raise BrokenPipeError('injected heartbeat write failure')
            original(handler, value)
        with patch.object(OpenAIHandler, '_sse', fail_heartbeat):
            response, _ = self.start(tools=True)
            self.assertTrue(self.entered.wait(1))
            self.assertTrue(failed.wait(0.5))
            self.release.set()
            self.assertEqual(response.read(), b'')
            self.assert_model_released()
        self.assertEqual(len(set(self.owner_threads)), 1)

    def test_chat_stream_disables_reverse_proxy_buffering(self):
        self.release.set()
        response, _ = self.start()
        self.assertEqual(response.getheader('X-Accel-Buffering'), 'no')
        self.finish(response)


class ChatEventsTests(unittest.TestCase):
    def test_terminal_events_are_atomic_and_stop_heartbeat_thread(self):
        from deepseek_v4_ssd.server import ChatEvents
        for failure in (False, True):
            with self.subTest(failure=failure):
                writes = []
                heartbeat = threading.Event()
                def send(value):
                    writes.append(value)
                    choices = value.get('choices', [])
                    if choices == [{'index': 0, 'delta': {}, 'finish_reason': None}]:
                        heartbeat.set()
                    if 'error' in value or any(c.get('finish_reason') for c in choices):
                        # Give the heartbeat writer time to contend for the terminal lock.
                        time.sleep(0.06)
                handler = SimpleNamespace(_sse=send, _sse_done=lambda: writes.append('[DONE]'))
                with patch('deepseek_v4_ssd.server.CHAT_HEARTBEAT_SECONDS', 0.01):
                    with ChatEvents(handler, {'id': 'chatcmpl-test'}) as events:
                        self.assertTrue(heartbeat.wait(0.5))
                        if failure:
                            events.end(APIError('Test error.', code='test_error').body())
                            self.assertIn('error', writes[-2])
                        else:
                            events.finish('stop', 5, 1, True)
                            self.assertEqual(writes[-3]['choices'][0]['finish_reason'], 'stop')
                            self.assertEqual(writes[-2]['usage']['completion_tokens'], 1)
                        self.assertEqual(writes[-1], '[DONE]')
                        count = len(writes)
                        events.send({'content': 'late'})
                        events.finish('stop', 0, 0, False)
                        time.sleep(0.03)
                        self.assertEqual(len(writes), count)
                    self.assertFalse(events.thread.is_alive())
                    self.assertEqual(writes.count('[DONE]'), 1)


if __name__ == '__main__':
    unittest.main()
