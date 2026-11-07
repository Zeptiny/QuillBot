"""MCP discovery, forwarding and failure isolation without a live service."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool as MCPTool

from core import radar_tools
from core.tools import ToolRegistry


def test_opt_in_registration(monkeypatch):
    registry = ToolRegistry()
    monkeypatch.setattr(radar_tools, 'RADAR_MCP_ENABLED', False)
    radar_tools.register(registry)
    assert registry.names() == []
    monkeypatch.setattr(radar_tools, 'RADAR_MCP_ENABLED', True)
    radar_tools.register(registry)
    assert registry.names() == list(radar_tools.RADAR_TOOLS)


async def test_protocol_discovery_and_call(monkeypatch):
    events = []

    @asynccontextmanager
    async def transport(url):
        events.append(('open', url))
        try:
            yield 'read', 'write', None
        finally:
            events.append('closed')

    class Session:
        def __init__(self, read, write):
            assert (read, write) == ('read', 'write')

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def initialize(self):
            events.append('initialize')

        async def list_tools(self, cursor=None):
            assert events[-1] == 'initialize'
            return ListToolsResult(tools=[MCPTool(
                name='buscar_processador', description='CPUs',
                inputSchema={'type': 'object', 'properties': {'preco_max': {'type': 'number'}}},
            )])

        async def call_tool(self, name, arguments):
            assert events[-1] == 'initialize'
            assert name == 'buscar_processador'
            assert arguments == {'preco_max': 1000}
            return CallToolResult(content=[TextContent(type='text', text='Sem resultados')], isError=True)

    monkeypatch.setattr(radar_tools, 'streamablehttp_client', transport)
    monkeypatch.setattr(radar_tools, 'ClientSession', Session)
    discovery = json.loads(await radar_tools._run('list', {}))
    assert discovery['tools'][0]['inputSchema']['properties']['preco_max']['type'] == 'number'
    result = json.loads(await radar_tools._run('call', {
        'name': 'buscar_processador', 'arguments': {'preco_max': 1000},
    }))
    assert result['isError'] is True
    assert result['content'][0]['text'] == 'Sem resultados'
    assert events.count('closed') == 2


async def test_failures_and_bad_arguments(monkeypatch):
    async def fail(*args):
        raise TimeoutError('unavailable')

    monkeypatch.setattr(radar_tools, '_request', fail)
    assert 'indisponível' in await radar_tools._run('list', {})
    assert 'Informe' in await radar_tools._run('call', {'name': 'x', 'arguments': []})
    assert 'Informe' in await radar_tools._run('call', {})


async def test_cancellation_propagates(monkeypatch):
    async def cancel(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(radar_tools, '_request', cancel)
    try:
        await radar_tools._run('list', {})
    except asyncio.CancelledError:
        return
    raise AssertionError('Cancellation swallowed')


async def test_timeout(monkeypatch):
    @asynccontextmanager
    async def stalled(url):
        await asyncio.sleep(10)
        yield SimpleNamespace()

    monkeypatch.setattr(radar_tools, 'RADAR_MCP_TIMEOUT_SECONDS', 0.01)
    monkeypatch.setattr(radar_tools, 'streamablehttp_client', stalled)
    assert 'indisponível' in await radar_tools._run('list', {})
