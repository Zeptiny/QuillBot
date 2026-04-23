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
    get_json as _get_json,
    search_hangar,
    search_modrinth,
    search_spiget,
)
from config import DOCS_BRANCH, GITHUB_API


# Basic hostname/IP pattern: letters, digits, dots, colons, hyphens; optional port
_IP_PATTERN = re.compile(r'^[\w.:-]+$')

logger = logging.getLogger(__name__)

MCSRVSTAT_API = 'https://api.mcsrvstat.us/3'


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


class PluginResultView(discord.ui.View):
    """Paginated view cycling through one plugin source per page."""

    def __init__(self, query: str, pages: list[tuple[str, str]]):
        super().__init__(timeout=120)
        self.query = query
        self.pages = pages
        self.index = 0
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        self.prev_btn.disabled = self.index == 0
        self.next_btn.disabled = self.index >= len(self.pages) - 1
        self.counter_btn.label = f'{self.index + 1}/{len(self.pages)}'

    def current_embed(self) -> discord.Embed:
        source_label, text = self.pages[self.index]
        embed = discord.Embed(
            title=f'🔌 Plugins: {self.query}',
            color=discord.Color.green(),
        )
        embed.add_field(name=source_label, value=text, inline=False)
        embed.set_footer(
            text=f'Página {self.index + 1} de {len(self.pages)} • Modrinth, Hangar e SpigotMC'
        )
        return embed

    @discord.ui.button(label='◄', style=discord.ButtonStyle.secondary)
    async def prev_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index -= 1
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.current_embed(), view=self)

    @discord.ui.button(label='1/1', style=discord.ButtonStyle.secondary, disabled=True)
    async def counter_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()

    @discord.ui.button(label='►', style=discord.ButtonStyle.secondary)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index += 1
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.current_embed(), view=self)


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

    # --- /plugin ---

    @app_commands.command(name='plugin', description='Pesquisa um plugin no Modrinth, Hangar e SpigotMC')
    @app_commands.describe(name='Nome do plugin para pesquisar')
    async def plugin_search(self, interaction: discord.Interaction, name: str):
        await interaction.response.defer(thinking=True)

        modrinth_results, hangar_results, spiget_results = await asyncio.gather(
            search_modrinth(self.session, name),
            search_hangar(self.session, name),
            search_spiget(self.session, name),
        )

        pages: list[tuple[str, str]] = []
        if modrinth_results:
            pages.append(('🟢 Modrinth', _format_plugin_list(modrinth_results)))
        if hangar_results:
            pages.append(('🔵 Hangar', _format_plugin_list(hangar_results)))
        if spiget_results:
            pages.append(('🟠 SpigotMC', _format_plugin_list(spiget_results)))

        if not pages:
            await interaction.followup.send(f'Nenhum plugin encontrado para `{name}`.')
            return

        view = PluginResultView(name, pages)
        await interaction.followup.send(embed=view.current_embed(), view=view)

    @plugin_search.autocomplete('name')
    async def _plugin_name_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        if not self.session or len(current) < 2:
            return []
        try:
            results = await search_modrinth(self.session, current)
            return [
                app_commands.Choice(name=r['name'][:100], value=r['name'][:100])
                for r in results[:6]
            ]
        except Exception:
            logger.exception('Autocomplete error for query %r', current)
            return []

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
        data = await _get_json(self.session, f'{MCSRVSTAT_API}/{safe_ip}')
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
        data = await _get_json(self.session, url)
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
