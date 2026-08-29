"""Smoke test for api_logger: both transports, redaction, body capture, error path."""

import asyncio
import json
import os
import tempfile

tmp = tempfile.mkdtemp()
os.environ['API_REQUEST_LOG_PATH'] = os.path.join(tmp, 'api.log')
os.environ['API_REQUEST_LOG_BODY'] = 'all'
os.environ['API_REQUEST_LOG_ENABLED'] = 'true'

import aiohttp
import httpx
from aiohttp import web

import api_logger

api_logger.install()


async def handler(request):
    await request.json()
    return web.json_response({'choices': [{'message': {'content': 'resposta'}}], 'usage': {'prompt_tokens': 42}})


async def main():
    app = web.Application()
    app.router.add_post('/chat', handler)
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

    errs = [l for l in lines if 'error' in l]
    assert len(errs) == 1 and errs[0]['url'].startswith('http://127.0.0.1:1'), 'error path line missing'

    meta = [l for l in lines if l['dir'] == 'meta']
    assert meta and meta[0]['event'] == 'api_request_logging_installed'

    inbound = [l for l in lines if l['dir'] == 'inbound']
    assert len(inbound) == 1 and inbound[0]['command'] == 'ask' and inbound[0]['user_id'] == 123
    assert inbound[0]['interaction_type'] == 'application_command'
    assert inbound[0]['guild_id'] == 456 and inbound[0]['channel_id'] == 789
    print('ALL_ASSERTS_PASSED')


asyncio.run(main())
