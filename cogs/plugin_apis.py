"""Shared plugin search helpers for Modrinth, Hangar and SpigotMC."""

import asyncio
import json
import logging
import urllib.parse

import aiohttp

logger = logging.getLogger(__name__)

MODRINTH_API = 'https://api.modrinth.com/v2'
HANGAR_API = 'https://hangar.papermc.io/api/v1'
SPIGET_API = 'https://api.spiget.org/v2'
HTTP_HEADERS = {'User-Agent': 'QuillBot/1.0 (github.com/Zeptiny/QuillBot)'}


async def get_json(
    session: aiohttp.ClientSession, url: str, **kwargs
) -> dict | list | None:
    """Fetch JSON from a URL, returning None on failure."""
    try:
        async with session.get(url, headers=HTTP_HEADERS, **kwargs) as resp:
            if resp.status == 200:
                return await resp.json()
            logger.warning("GET %s returned %d", url, resp.status)
    except Exception:
        logger.exception("Failed to fetch %s", url)
    return None


async def search_modrinth(session: aiohttp.ClientSession, query: str) -> list[dict]:
    params = {
        'query': query,
        'limit': 3,
        'facets': json.dumps([['project_type:plugin']]),
    }
    data = await get_json(session, f'{MODRINTH_API}/search', params=params)
    if not data:
        return []
    results = []
    for hit in data.get('hits', [])[:3]:
        game_versions = hit.get('versions', [])
        slug_or_id = hit.get('slug') or hit.get('project_id', '')
        results.append({
            'name': hit.get('title', '?'),
            'author': hit.get('author', '?'),
            'description': hit.get('description', '')[:100],
            'downloads': hit.get('downloads', 0),
            'url': f'https://modrinth.com/project/{slug_or_id}',
            'versions': game_versions[-3:] if game_versions else [],
            'source': 'Modrinth',
        })
    return results


async def search_hangar(session: aiohttp.ClientSession, query: str) -> list[dict]:
    params = {'q': query, 'limit': 3}
    data = await get_json(session, f'{HANGAR_API}/projects', params=params)
    if not data:
        return []
    results = []
    for project in data.get('result', [])[:3]:
        ns = project.get('namespace', {})
        owner = ns.get('owner', '?')
        slug = ns.get('slug', '')
        results.append({
            'name': project.get('name', '?'),
            'author': owner,
            'description': project.get('description', '')[:100],
            'downloads': project.get('stats', {}).get('downloads', 0),
            'url': f'https://hangar.papermc.io/{owner}/{slug}',
            'versions': [],
            'source': 'Hangar',
        })
    return results


async def search_spiget(session: aiohttp.ClientSession, query: str) -> list[dict]:
    encoded = urllib.parse.quote(query)
    data = await get_json(
        session,
        f'{SPIGET_API}/search/resources/{encoded}'
        '?sort=-downloads&size=3&fields=id,name,downloads,testedVersions',
    )
    if not isinstance(data, list):
        return []
    results = []
    for res in data[:3]:
        res_id = res.get('id', '')
        results.append({
            'name': res.get('name', '?'),
            'author': '?',
            'description': '',
            'downloads': res.get('downloads', 0),
            'url': f'https://www.spigotmc.org/resources/{res_id}',
            'versions': res.get('testedVersions', [])[:3],
            'source': 'SpigotMC',
        })
    return results


async def search_all(session: aiohttp.ClientSession, query: str) -> list[dict]:
    """Search Modrinth, Hangar, and SpigotMC concurrently."""
    modrinth, hangar, spiget = await asyncio.gather(
        search_modrinth(session, query),
        search_hangar(session, query),
        search_spiget(session, query),
    )
    return modrinth + hangar + spiget
