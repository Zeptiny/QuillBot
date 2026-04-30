"""Tavily web search and content extraction tools for the LLM agentic loop."""

import json
import logging
from typing import Any

from tavily import TavilyClient

from config import TAVILY_API_KEY, TAVILY_AVAILABLE

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6

WEB_SEARCH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'web_search',
        'description': (
            'Pesquisa na web para informações em tempo real ou recentes. '
            'Use quando precisar de dados atualizados, notícias, preços, versões ou qualquer informação '
            'que pode ter mudado após seu treinamento. '
            'Para buscas gerais, use search_depth="basic". '
            'Para buscas mais detalhadas e precisas, use search_depth="advanced". '
            'Quando citar resultados, inclua o título e o link da fonte.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'A consulta de busca em linguagem natural.',
                },
                'search_depth': {
                    'type': 'string',
                    'enum': ['basic', 'advanced'],
                    'description': (
                        'Profundidade da busca. Use "basic" para buscas rápidas e gerais, '
                        '"advanced" para buscas mais detalhadas e precisas.'
                    ),
                },
                'max_results': {
                    'type': 'integer',
                    'description': (
                        'Número máximo de resultados (padrão: 5, intervalo: 1-10). '
                        'Use 3 para perguntas focadas, 5-10 para tópicos amplos.'
                    ),
                },
                'time_range': {
                    'type': 'string',
                    'enum': ['day', 'week', 'month', 'year'],
                    'description': (
                        'Período de tempo para filtrar resultados. '
                        'Use "day" ou "week" para notícias muito recentes, '
                        '"month" para tópicos sazonais, omita para sem filtro.'
                    ),
                },
                'include_domains': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Lista de domínios para INCLUIR nos resultados (ex: ["minecraft.wiki", "papermc.io"]). '
                        'Use quando quiser restringir a busca a fontes específicas.'
                    ),
                },
                'exclude_domains': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Lista de domínios para EXCLUIR dos resultados (ex: ["reddit.com", "twitter.com"]). '
                        'Use para evitar fontes não confiáveis ou irrelevantes.'
                    ),
                },
            },
            'required': ['query'],
        },
    },
}

WEB_EXTRACT_TOOL = {
    'type': 'function',
    'function': {
        'name': 'web_extract',
        'description': (
            'Extrai o conteúdo de URLs específicas para leitura detalhada. '
            'Use quando os resultados de web_search forem promissores mas insuficientes, '
            'ou quando precisar do conteúdo completo de uma página. '
            'Sempre prefira web_search primeiro; use web_extract para se aprofundar em fontes relevantes.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'urls': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Lista de URLs para extrair conteúdo (máximo 5). '
                        'Use URLs retornadas pelo web_search ou fornecidas pelo usuário.'
                    ),
                },
                'query': {
                    'type': 'string',
                    'description': (
                        'Consulta opcional para filtrar o conteúdo extraído pela relevância. '
                        'Quando fornecida, o conteúdo será priorizado pelas seções relevantes à consulta.'
                    ),
                },
            },
            'required': ['urls'],
        },
    },
}

TOOLS = [WEB_SEARCH_TOOL, WEB_EXTRACT_TOOL]


def _get_tavily_client() -> TavilyClient | None:
    if not TAVILY_API_KEY:
        return None
    try:
        return TavilyClient(api_key=TAVILY_API_KEY)
    except Exception:
        logger.exception("Failed to initialize Tavily client")
        return None


async def exec_web_search(args: dict) -> tuple[str, list[dict]]:
    """Execute a web_search tool call via Tavily API.

    Returns (result_text, source_list) where source_list contains
    dicts with 'title', 'url', and 'content' keys for citation.
    """
    client = _get_tavily_client()
    if client is None:
        return 'Busca web indisponível: TAVILY_API_KEY não configurada.', []

    query = args.get('query', '')
    if not query:
        return 'Consulta de busca vazia.', []

    kwargs: dict[str, Any] = {
        'query': query,
        'max_results': max(1, min(10, int(args.get('max_results', 5)))),
    }

    search_depth = args.get('search_depth')
    if search_depth in ('basic', 'advanced'):
        kwargs['search_depth'] = search_depth

    time_range = args.get('time_range')
    if time_range in ('day', 'week', 'month', 'year'):
        kwargs['time_range'] = time_range

    include_domains = args.get('include_domains')
    if include_domains and isinstance(include_domains, list):
        kwargs['include_domains'] = include_domains

    exclude_domains = args.get('exclude_domains')
    if exclude_domains and isinstance(exclude_domains, list):
        kwargs['exclude_domains'] = exclude_domains

    try:
        response = client.search(**kwargs)
    except Exception:
        logger.exception("Tavily search failed for query: %s", query[:80])
        return 'Erro ao realizar busca web. Tente novamente.', []

    results = response.get('results', [])
    if not results:
        return 'Nenhum resultado encontrado na web.', []

    parts = []
    sources = []
    for r in results:
        title = r.get('title', 'Sem título')
        url = r.get('url', '')
        content = r.get('content', '')
        score = r.get('score', 0)
        parts.append(f"**{title}** ({url})\n{content}")
        sources.append({'title': title, 'url': url, 'content': content, 'score': score})

    result_text = '\n\n---\n\n'.join(parts)
    return result_text, sources


async def exec_web_extract(args: dict) -> tuple[str, list[dict]]:
    """Execute a web_extract tool call via Tavily API.

    Returns (result_text, source_list) where source_list contains
    dicts with 'title', 'url', and 'content' keys for citation.
    """
    client = _get_tavily_client()
    if client is None:
        return 'Extração web indisponível: TAVILY_API_KEY não configurada.', []

    urls = args.get('urls', [])
    if not urls:
        return 'Nenhuma URL fornecida para extração.', []

    urls = urls[:5]

    kwargs: dict[str, Any] = {
        'urls': urls,
    }

    query = args.get('query')
    if query:
        kwargs['query'] = query

    try:
        response = client.extract(**kwargs)
    except Exception:
        logger.exception("Tavily extract failed for URLs: %s", urls)
        return 'Erro ao extrair conteúdo das URLs. Tente novamente.', []

    results = response.get('results', [])
    failed = response.get('failed_results', [])

    if not results and not failed:
        return 'Nenhum conteúdo extraído das URLs fornecidas.', []

    parts = []
    sources = []
    for r in results:
        url = r.get('url', '')
        content = r.get('raw_content', r.get('content', ''))
        if not content:
            continue
        if len(content) > 4000:
            content = content[:4000] + '\n\n[... conteúdo truncado ...]'
        parts.append(f"**Conteúdo extraído de:** {url}\n\n{content}")
        sources.append({'title': url, 'url': url, 'content': content})

    if failed:
        failed_urls = [f.get('url', '?') for f in failed]
        parts.append(f"**Falha ao extrair:** {', '.join(failed_urls)}")

    if not parts:
        return 'Nenhum conteúdo extraído das URLs fornecidas.', []

    result_text = '\n\n---\n\n'.join(parts)
    return result_text, sources


async def exec_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    """Dispatch a tool call by name. Returns (result_text, sources)."""
    if name == 'web_search':
        return await exec_web_search(args)
    if name == 'web_extract':
        return await exec_web_extract(args)
    return f'Ferramenta desconhecida: {name}', []


def status_label(name: str, args: dict) -> str:
    """Return a short Portuguese status string shown while a tool runs."""
    if name == 'web_search':
        query = args.get('query', '')
        depth = args.get('search_depth', 'basic')
        label = f'🌐 Buscando na web: *{query[:60]}*'
        if depth == 'advanced':
            label += ' (busca avançada)'
        return label
    if name == 'web_extract':
        urls = args.get('urls', [])
        url_list = ', '.join(str(u)[:40] for u in urls[:3])
        return f'📄 Extraindo conteúdo de: {url_list}'
    return f'🔧 Executando: {name}'