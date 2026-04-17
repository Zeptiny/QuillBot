import asyncio
import logging
import re
import urllib.parse

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from cogs.plugin_apis import (
    HTTP_HEADERS as _HEADERS,
    search_hangar,
    search_modrinth,
    search_spiget,
)
from config import DOCS_BRANCH, GITHUB_API


# Basic hostname/IP pattern: letters, digits, dots, colons, hyphens; optional port
_IP_PATTERN = re.compile(r'^[\w.:-]+$')

logger = logging.getLogger(__name__)

MCSRVSTAT_API = 'https://api.mcsrvstat.us/3'


class PluginSearch(commands.Cog):
    """Plugin search, server status and docs changelog commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.session: aiohttp.ClientSession | None = None

    async def cog_load(self):
        self.session = aiohttp.ClientSession(
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        )

    async def cog_unload(self):
        if self.session:
            await self.session.close()

    async def _get_json(self, url: str, **kwargs) -> dict | list | None:
        """Fetch JSON from a URL, returning None on failure."""
        try:
            async with self.session.get(url, **kwargs) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning("GET %s returned %d", url, resp.status)
        except Exception:
            logger.exception("Failed to fetch %s", url)
        return None

    # --- /plugin ---

    @staticmethod
    def _format_plugin_list(results: list[dict]) -> str:
        lines = []
        for r in results:
            versions = ', '.join(str(v) for v in r['versions'][:3]) if r['versions'] else '—'
            desc = r['description']
            line = (
                f"**[{r['name']}]({r['url']})** por `{r['author']}`\n"
                f"⬇️ `{r['downloads']:,}` downloads | 🎮 `{versions}`"
            )
            if desc:
                line += f"\n{desc}"
            lines.append(line)
        return '\n\n'.join(lines) if lines else 'Sem resultados.'

    @app_commands.command(name='plugin', description='Pesquisa um plugin no Modrinth, Hangar e SpigotMC')
    @app_commands.describe(name='Nome do plugin para pesquisar')
    async def plugin_search(self, interaction: discord.Interaction, name: str):
        await interaction.response.defer(thinking=True)

        modrinth_results, hangar_results = await asyncio.gather(
            search_modrinth(self.session, name),
            search_hangar(self.session, name),
        )

        secondary_results = hangar_results
        if not modrinth_results and not secondary_results:
            # Fallback to Spiget
            spiget_results = await search_spiget(self.session, name)
            if not spiget_results:
                await interaction.followup.send(
                    f'Nenhum plugin encontrado para `{name}`.'
                )
                return
            secondary_results = spiget_results

        embed = discord.Embed(
            title=f'🔌 Plugins encontrados: {name}',
            color=discord.Color.green(),
        )

        if modrinth_results:
            embed.add_field(
                name='🟢 Modrinth',
                value=self._format_plugin_list(modrinth_results),
                inline=False,
            )

        if secondary_results:
            source_name = secondary_results[0].get('source', 'Hangar')
            embed.add_field(
                name=f'🔵 {source_name}',
                value=self._format_plugin_list(secondary_results),
                inline=False,
            )

        embed.set_footer(text='Dados do Modrinth, Hangar e SpigotMC')
        await interaction.followup.send(embed=embed)

    # --- /status ---

    @app_commands.command(name='status', description='Verifica o status de um servidor Minecraft')
    @app_commands.describe(ip='Endereço IP ou hostname do servidor')
    async def server_status(self, interaction: discord.Interaction, ip: str):
        if not _IP_PATTERN.match(ip):
            await interaction.response.send_message(
                'Endereço inválido. Use um IP ou hostname válido (ex: `mc.example.com` ou `1.2.3.4:25565`).',
                ephemeral=True,
            )
            return
        await interaction.response.defer(thinking=True)

        safe_ip = urllib.parse.quote(ip, safe='')
        data = await self._get_json(f'{MCSRVSTAT_API}/{safe_ip}')
        if data is None:
            await interaction.followup.send(
                'Não foi possível consultar o servidor. Verifique o IP e tente novamente.'
            )
            return

        online = data.get('online', False)

        if not online:
            embed = discord.Embed(
                title='🔴 Servidor Offline',
                description=f'`{ip}` está offline ou não foi encontrado.',
                color=discord.Color.red(),
            )
            await interaction.followup.send(embed=embed)
            return

        players = data.get('players', {})
        online_count = players.get('online', 0)
        max_count = players.get('max', 0)
        version = data.get('version', '?')
        motd_lines = data.get('motd', {}).get('clean', [])
        motd = '\n'.join(motd_lines) if motd_lines else ''

        embed = discord.Embed(
            title=f'🟢 Servidor Online: `{ip}`',
            color=discord.Color.green(),
        )
        if motd:
            embed.description = motd
        embed.add_field(name='👥 Jogadores', value=f'{online_count}/{max_count}', inline=True)
        embed.add_field(name='🎮 Versão', value=version, inline=True)
        embed.set_footer(text='Dados fornecidos por mcsrvstat.us')
        await interaction.followup.send(embed=embed)

    # --- /changelog ---

    @app_commands.command(
        name='changelog',
        description="Exibe os commits mais recentes da documentação do Miners' Refuge",
    )
    async def changelog(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)

        url = f'{GITHUB_API}/commits?sha={DOCS_BRANCH}&per_page=5'
        data = await self._get_json(url)
        if not data or not isinstance(data, list):
            await interaction.followup.send(
                'Não foi possível obter o changelog da documentação.'
            )
            return

        embed = discord.Embed(
            title='📋 Changelog da Documentação',
            color=discord.Color.blurple(),
        )

        lines = []
        for entry in data[:5]:
            sha = entry.get('sha', '')[:7]
            commit = entry.get('commit', {})
            message = commit.get('message', '').split('\n')[0][:80]
            author = commit.get('author', {}).get('name', '?')
            date = commit.get('author', {}).get('date', '')[:10]
            html_url = entry.get('html_url', '')
            lines.append(f'[`{sha}`]({html_url}) **{message}**\n> {author} • {date}')

        embed.description = '\n\n'.join(lines)
        embed.set_footer(text='MinersRefuge/docs')
        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(PluginSearch(bot))
