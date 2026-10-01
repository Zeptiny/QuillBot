"""api_logger: both transports, redaction, body capture, error path, inbound hook.

install() patches aiohttp, httpx and httpx2 for the whole process and config reads the
log settings at import, so the scenario runs in its own interpreter: pytest
collects test_api_logger(), which re-runs this file as a script.
"""

import asyncio
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_api_logger(tmp_path):
    env = dict(
        os.environ,
        API_REQUEST_LOG_PATH=str(tmp_path / 'api.log'),
        API_REQUEST_LOG_BODY='all',
        API_REQUEST_LOG_ENABLED='true',
        PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get('PYTHONPATH')])),
    )
    proc = subprocess.run([sys.executable, __file__], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


async def handler(request):
    await request.json()
    return web.json_response({'choices': [{'message': {'content': 'resposta'}}], 'usage': {'prompt_tokens': 42}})


SDK_BODIES = []


def _sse(*chunks):
    return ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + 'data: [DONE]\n\n'


async def completions_handler(request):
    """Streaming-only, like zai behind the gateway: plain requests get its 400."""
    body = await request.json()
    SDK_BODIES.append(body)
    if not body.get('stream'):
        return web.json_response({'error': {
            'type': 'provider_error', 'code': 'streaming_only', 'param': 'stream',
            'message': 'Only streaming requests are supported. Set stream to true.',
        }}, status=400)
    head = {'id': 'c1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm'}

    def chunk(delta, finish_reason=None):
        return {**head, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]}

    usage = {**head, 'choices': [], 'usage': {
        'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10,
        'prompt_tokens_details': {'cached_tokens': 4},
    }}
    if body.get('tools') and len(body['messages']) == 1:
        events = _sse(
            chunk({'role': 'assistant', 'reasoning_content': 'pensando '}),
            chunk({'reasoning_content': 'mais'}),
            chunk({'tool_calls': [{'index': 0, 'id': 'call_a', 'type': 'function',
                                   'function': {'name': 'web_search', 'arguments': '{"q": '}}]}),
            chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '"lag"}'}}]}),
            chunk({}, 'tool_calls'),
            usage,
        )
    else:
        events = _sse(
            chunk({'role': 'assistant', 'content': 'o'}),
            chunk({'content': 'k'}),
            chunk({}, 'stop'),
            usage,
        )
    return web.Response(text=events, content_type='text/event-stream')


async def main():
    app = web.Application()
    app.router.add_post('/chat', handler)
    app.router.add_post('/v1/chat/completions', completions_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    base = f'http://127.0.0.1:{runner.addresses[0][1]}'

    async with aiohttp.ClientSession() as s:
        async with s.post(base + '/chat?token=supersecret&x=1', json={'q': 'a'}) as r:
            assert (await r.json())['usage']['prompt_tokens'] == 42

    async with httpx.AsyncClient() as c:
        r = await c.post(base + '/chat', json={'model': 'qwen/qwen3.6-plus', 'messages': [{'role': 'user', 'content': 'oi'}]})
        assert r.status_code == 200
        assert r.json()['choices'][0]['message']['content'] == 'resposta'  # body still readable by the caller
        try:
            await c.get('http://127.0.0.1:1/nope')
        except Exception:
            pass

    # The real OpenAI SDK: openai>=2 sends through httpx2, not httpx. The bot
    # only streams (create_chat_completion); the fake endpoint rejects anything else.
    from openai import AsyncOpenAI, BadRequestError

    from cogs.utils import create_chat_completion, serialize_trajectory
    client = AsyncOpenAI(base_url=base + '/v1', api_key='test-key', max_retries=0)
    try:
        await client.chat.completions.create(model='sdk/model', messages=[{'role': 'user', 'content': 'x'}])
        raise AssertionError('non-streaming request should have been rejected')
    except BadRequestError as e:
        assert 'streaming_only' in str(e)
    SDK_BODIES.clear()

    messages = [{'role': 'user', 'content': 'via sdk'}]
    tools = [{'type': 'function', 'function': {'name': 'web_search', 'parameters': {'type': 'object'}}}]
    first = await create_chat_completion(client, model='sdk/model', messages=messages, max_tokens=50, tools=tools)
    msg = first.choices[0].message
    assert first.choices[0].finish_reason == 'tool_calls'
    assert msg.content is None
    assert [(tc.id, tc.function.name, tc.function.arguments) for tc in msg.tool_calls] == [('call_a', 'web_search', '{"q": "lag"}')]
    assert msg.reasoning_content == 'pensando mais'
    assert first.usage.prompt_tokens == 7 and first.usage.prompt_tokens_details.cached_tokens == 4
    assert SDK_BODIES[0]['stream'] is True and SDK_BODIES[0]['stream_options'] == {'include_usage': True}
    assert SDK_BODIES[0]['messages'] == [{'role': 'user', 'content': 'via sdk'}]

    # The assembled message replays like the SDK's own: same fields on the wire.
    messages.append(msg)
    messages.append({'role': 'tool', 'tool_call_id': 'call_a', 'content': 'r'})
    done = await create_chat_completion(client, model='sdk/model', messages=messages, max_tokens=50, tools=tools)
    assert done.choices[0].message.content == 'ok' and done.choices[0].finish_reason == 'stop'
    sent = SDK_BODIES[1]['messages'][1]
    assert sent == {
        'role': 'assistant', 'content': None, 'reasoning_content': 'pensando mais',
        'tool_calls': [{'id': 'call_a', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"q": "lag"}'}}],
    }, sent
    assert serialize_trajectory([msg])[0]['tool_calls'] == sent['tool_calls']
    await client.close()

    await runner.cleanup()

    import discord
    from unittest.mock import MagicMock
    from discord.ext import commands

    assert api_logger._service('https://openrouter.ai/api/v1/chat/completions') == 'openai'
    assert api_logger._service('https://api.tavily.com/search') == 'tavily'
    assert api_logger._service('https://api.github.com/repos/x/y') == 'github'
    assert api_logger._service('https://discord.com/api/v10/channels/1') == 'discord'

    bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())
    api_logger.install_inbound_hooks(bot)
    inter = MagicMock()
    inter.type = discord.InteractionType.application_command
    inter.data = {'name': 'ask'}
    inter.user.id = 123
    inter.guild_id = 456
    inter.channel_id = 789
    # Client.dispatch needs a running loop (discord.py >=2.7), so invoke the
    # registered listener directly — same coroutine the event loop would run.
    await bot.extra_events['on_interaction'][0](inter)

    lines = [json.loads(l) for l in open(os.environ['API_REQUEST_LOG_PATH'])]
    for l in lines:
        print(json.dumps(l, ensure_ascii=False))

    aiohttp_lines = [l for l in lines if l['dir'] == 'outbound' and l['url'].startswith(base) and l.get('status') == 200 and 'token' in l['url']]
    assert len(aiohttp_lines) == 1, 'aiohttp success line missing'
    assert 'supersecret' not in aiohttp_lines[0]['url'] and 'REDACTED' in aiohttp_lines[0]['url'], 'query redaction failed'
    assert aiohttp_lines[0]['service'] == 'other' and aiohttp_lines[0]['status'] == 200 and 'duration_ms' in aiohttp_lines[0]

    hx = [l for l in lines if l['dir'] == 'outbound' and l['method'] == 'POST' and l['url'].endswith('/chat')]
    assert len(hx) == 1 and hx[0]['status'] == 200, 'httpx success line missing'
    assert hx[0]['model'] == 'qwen/qwen3.6-plus', 'model extraction failed'
    assert hx[0]['request_body']['messages'][0]['content'] == 'oi', 'request body capture failed'
    assert hx[0]['response_body']['usage']['prompt_tokens'] == 42, 'response body capture failed'

    sdk = [l for l in lines if l['dir'] == 'outbound' and l['url'].endswith('/v1/chat/completions')]
    assert len(sdk) == 3, 'OpenAI SDK request not logged'
    assert sdk[0]['status'] == 400 and sdk[0]['response_body']['error']['code'] == 'streaming_only'
    assert sdk[1]['status'] == 200 and sdk[1]['request_body']['messages'][0]['content'] == 'via sdk', 'SDK request body capture failed'
    assert sdk[1]['model'] == 'sdk/model'
    streamed = sdk[1]['response_body']
    assert streamed['reasoning_content'] == 'pensando mais' and streamed['finish_reason'] == 'tool_calls'
    assert streamed['tool_calls'] == [{'id': 'call_a', 'name': 'web_search', 'arguments': '{"q": "lag"}'}]
    assert streamed['usage']['prompt_tokens'] == 7 and streamed['stream_chunks'] == 6
    assert sdk[2]['response_body']['content'] == 'ok'
    assert 'test-key' not in json.dumps(sdk), 'API key leaked into the log'

    # Bodies: inline images shrink to their size; long bodies keep both ends.
    b64 = 'A' * 5000
    shrunk = api_logger._body_value(json.dumps({'u': f'data:image/png;base64,{b64}'}))
    assert shrunk == {'u': 'data:image/png;base64,<5000 chars>'}, shrunk
    limit = api_logger.API_REQUEST_LOG_BODY_MAX_CHARS
    long_body = 'HEAD' + 'x' * (limit * 2) + 'TAIL'
    cut = api_logger._body_value(long_body)
    assert cut.startswith('HEAD') and cut.endswith('TAIL') and 'truncated' in cut
    assert len(cut) < limit + 100

    errs = [l for l in lines if 'error' in l]
    assert len(errs) == 1 and errs[0]['url'].startswith('http://127.0.0.1:1'), 'error path line missing'

    meta = [l for l in lines if l['dir'] == 'meta']
    assert meta and meta[0]['event'] == 'api_request_logging_installed'

    inbound = [l for l in lines if l['dir'] == 'inbound']
    assert len(inbound) == 1 and inbound[0]['command'] == 'ask' and inbound[0]['user_id'] == 123
    assert inbound[0]['interaction_type'] == 'application_command'
    assert inbound[0]['guild_id'] == 456 and inbound[0]['channel_id'] == 789


if __name__ == '__main__':
    import aiohttp
    import httpx
    from aiohttp import web

    import api_logger

    api_logger.install()
    asyncio.run(main())
