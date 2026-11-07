"""Optional Radar de Hardware MCP bridge, using the official MCP client."""

import asyncio
import json
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import RADAR_MCP_ENABLED, RADAR_MCP_TIMEOUT_SECONDS, RADAR_MCP_URL
from core.tools import Tool, ToolRegistry, tool_schema

logger = logging.getLogger(__name__)

RADAR_TOOLS = ('radar_list_tools', 'radar_call_tool')


async def _request(method: str, args: dict) -> str:
    # A fresh initialized session per request also works with stateless servers;
    # context managers close the transport on success, failure and cancellation.
    async with asyncio.timeout(RADAR_MCP_TIMEOUT_SECONDS):
        async with streamablehttp_client(RADAR_MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if method == 'list':
                    tools = []
                    cursor = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        tools.extend(tool.model_dump(mode='json', exclude_none=True) for tool in page.tools)
                        cursor = page.nextCursor
                        if not cursor:
                            break
                    return json.dumps({'tools': tools}, ensure_ascii=False)
                result = await session.call_tool(args['name'], arguments=args.get('arguments', {}))
                # Preserve isError, structuredContent and resource links as well
                # as text, so a remote tool failure is never reported as success.
                return result.model_dump_json(exclude_none=True)


async def _run(method: str, args: dict) -> str:
    if method == 'call' and (
        not isinstance(args.get('name'), str) or not args['name'].strip()
        or not isinstance(args.get('arguments', {}), dict)
    ):
        return 'Informe name e arguments (objeto) conforme radar_list_tools.'
    try:
        return await _request(method, args)
    except Exception:
        logger.exception('Radar de Hardware MCP request failed')
        return 'Radar de Hardware indisponível no momento. Tente novamente; não invente preços ou resultados.'


def register(registry: ToolRegistry) -> None:
    if not RADAR_MCP_ENABLED:
        return
    registry.register(
        Tool(
            RADAR_TOOLS[0],
            tool_schema(
                RADAR_TOOLS[0],
                'Lista ferramentas e schemas do Radar de Hardware para consultar preços atuais, '
                'histórico e especificações de GPUs, CPUs e RAM no Brasil. '
                'Use antes de radar_call_tool para descobrir os nomes e argumentos disponíveis.',
            ),
            lambda ctx, args: _run('list', args),
            '🔎 Consultando ferramentas do Radar de Hardware',
        ),
        Tool(
            RADAR_TOOLS[1],
            tool_schema(
                RADAR_TOOLS[1],
                'Executa uma consulta no MCP Radar de Hardware. Use somente nomes e argumentos '
                'obtidos por radar_list_tools. Cite os links retornados e diferencie preços atuais '
                'de mínimos históricos; não invente dados quando houver erro.',
                {
                    'name': {'type': 'string', 'description': 'Nome retornado por radar_list_tools.'},
                    'arguments': {'type': 'object', 'description': 'Argumentos conforme o inputSchema da ferramenta.'},
                },
                ['name', 'arguments'],
            ),
            lambda ctx, args: _run('call', args),
            '🖥️ Consultando o Radar de Hardware',
        ),
    )
