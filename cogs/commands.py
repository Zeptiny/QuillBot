"""Utility slash commands: /docs, /help, /flags, /plov, /plano, /health, /sync.

Conversations (``/chat``, ``/ask``, mentions, replies) live in ``cogs/chat.py``.
"""

import asyncio
import datetime
import logging
import time

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from config import DOCS_BASE_URL, TAVILY_AVAILABLE, WEB_SEARCH_ENABLED

logger = logging.getLogger(__name__)


class Commands(commands.Cog):
    """Slash commands for server administration help."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self._start_time: float = time.monotonic()

    @app_commands.command(name="plov", description="Informações necessárias para escolher um serviço de hospedagem")
    async def hosting_info(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="PLOV - Informações para Hospedagem",
            color=discord.Color.blue(),
        )
        embed.add_field(name="P - Plano/Players", value="Quais recursos procura", inline=False)
        embed.add_field(name="L - Localização", value="Onde", inline=False)
        embed.add_field(name="O - Orçamento", value="Quanto está disposto a pagar", inline=False)
        embed.add_field(name="V - Versão", value="Qual a versão do servidor", inline=False)
        embed.set_footer(text="Se está em dúvida em qual plano precise, use /plano")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="plano", description="Informações para recomendar um plano adequado")
    async def plan_info(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Informações para Recomendação de Plano",
            color=discord.Color.green(),
        )
        embed.add_field(name="Versão", value="Qual a versão do servidor", inline=False)
        embed.add_field(name="Players", value="Quantidade de players simultâneos", inline=False)
        embed.add_field(name="Mods/Plugins", value="Quantos mods/plugins (Especifique)", inline=False)
        embed.add_field(name="Modo", value="Qual o modo de jogo do server", inline=False)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="docs", description="Pesquisar ou acessar a documentação do Miners' Refuge")
    @app_commands.describe(query="Termo para pesquisar na documentação (opcional)")
    async def docs_search(self, interaction: discord.Interaction, query: str | None = None):
        if query:
            # Delegate to RAG search if available
            docs_rag = self.services.docs
            if docs_rag is not None and docs_rag.indexing:
                await interaction.response.send_message(
                    '📚 A documentação está sendo indexada. Tente novamente em alguns instantes.',
                    ephemeral=True,
                )
                return
            if docs_rag is not None and docs_rag.chunk_count:
                await interaction.response.defer(thinking=True)
                try:
                    await interaction.edit_original_response(content=f'🔍 Pesquisando documentação: *{query[:60]}*')
                except discord.HTTPException:
                    pass
                results = await docs_rag.search(query, top_k=5)
                try:
                    await interaction.edit_original_response(content=None)
                except discord.HTTPException:
                    pass
                if results:
                    embed = discord.Embed(
                        title=f"🔍 Resultados para: {query}",
                        color=discord.Color.gold(),
                    )
                    seen_paths = set()
                    lines = []
                    for r in results:
                        if r['path'] not in seen_paths:
                            seen_paths.add(r['path'])
                            url = r.get('url') or r.get('doc_url', '')
                            title = r['title'] or r['path']
                            source = r.get('source')
                            # Show a snippet of the content
                            snippet = r['content'][:120].replace('\n', ' ').strip()
                            source_prefix = f'`{source}` ' if source else ''
                            lines.append(f'**[{title}]({url})**\n{source_prefix}{snippet}…')
                    embed.description = '\n\n'.join(lines)
                    embed.set_footer(text=f'Use /ask para perguntas detalhadas • {DOCS_BASE_URL}')
                    await interaction.followup.send(embed=embed)
                    return
            # Fallback: no RAG available
            embed = discord.Embed(
                title="Documentação - Miners' Refuge",
                color=discord.Color.gold(),
                description=(
                    f"Pesquise por **{query}** na documentação:\n"
                    f"[Abrir documentação]({DOCS_BASE_URL})\n\n"
                    f"Use `CTRL+K` no site para pesquisar diretamente!"
                ),
            )
            embed.set_footer(text="Contribua abrindo um PR no GitHub!")
            if interaction.response.is_done():
                await interaction.followup.send(embed=embed)
            else:
                await interaction.response.send_message(embed=embed)
        else:
            embed = discord.Embed(
                title="Documentação - Miners' Refuge",
                color=discord.Color.gold(),
                description=(
                    f"Acesse a documentação completa:\n"
                    f"[docs.minersrefuge.com.br]({DOCS_BASE_URL})\n\n"
                    f"Encontre guias sobre administração de servidores Minecraft, "
                    f"dicas de otimização e muito mais!"
                ),
            )
            embed.set_footer(text="Contribua abrindo um PR no GitHub!")
            await interaction.response.send_message(embed=embed)

    @app_commands.command(name="help", description="Lista todos os comandos disponíveis do bot")
    async def help_command(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="📋 Comandos do QuillBot",
            description="Todos os comandos disponíveis:",
            color=discord.Color.purple(),
        )

        # Dynamically list all registered slash commands
        cmds = self.bot.tree.get_commands()
        cmd_lines = []
        # Action GIF commands (/hug, /slap…) share one line in their own field.
        actions = sorted(c.name for c in cmds if getattr(c, 'extras', {}).get('action'))
        for cmd in sorted(cmds, key=lambda c: c.name):
            if getattr(cmd, 'extras', {}).get('action'):
                continue
            params = ''
            if hasattr(cmd, 'parameters') and cmd.parameters:
                param_parts = []
                for p in cmd.parameters:
                    if p.required:
                        param_parts.append(f'`{p.name}`')
                    else:
                        param_parts.append(f'`[{p.name}]`')
                params = ' ' + ' '.join(param_parts)
            cmd_lines.append(f'**/{cmd.name}**{params} — {cmd.description}')

        embed.description = '\n'.join(cmd_lines)
        if actions:
            embed.add_field(
                name="🎭 Ações",
                value=(
                    ' '.join(f'/{name}' for name in actions)
                    + "\nUm GIF de anime de você fazendo isso com alguém (`usuario`), que pode retribuir. "
                    "Pode fazer comigo também — e eu posso fazer com você."
                ),
                inline=False,
            )

        embed.add_field(
            name="💬 Conversa",
            value=(
                "Me mencione, responda a uma mensagem minha ou me chame pelo nome — "
                "várias mensagens seguidas viram uma resposta só. Conversas longas vão para "
                "uma thread; apagar ou editar sua mensagem logo depois apaga ou refaz a resposta.\n"
                "Admins ajustam por canal com `/config` e definem minha personalidade com `/persona`."
            ),
            inline=False,
        )
        embed.add_field(
            name="📝 Detecção Automática",
            value=(
                "O bot também analisa automaticamente logs e erros enviados no chat:\n"
                "• Links do **mclo.gs** e **pastebin.com**\n"
                "• Arquivos **.log** e **.txt** anexados\n"
                "• Mensagens com erros conhecidos de Minecraft"
            ),
            inline=False,
        )
        embed.set_footer(text=f"Miners' Refuge • {DOCS_BASE_URL}")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="flags", description="Gera as flags JVM de Aikar para o seu servidor Minecraft")
    @app_commands.describe(ram="Quantidade de RAM em MB (ex: 4096 para 4 GB)")
    async def jvm_flags(self, interaction: discord.Interaction, ram: int):
        if ram < 512:
            await interaction.response.send_message(
                'Recomendamos pelo menos 512 MB de RAM. Informe o valor em MB '
                '(ex.: 4096 para 4 GB).',
                ephemeral=True,
            )
            return
        if ram > 65536:
            await interaction.response.send_message(
                'Valor muito alto. Insira a quantidade em MB (ex: 4096 para 4 GB).',
                ephemeral=True,
            )
            return

        # Aikar's flags — adjusted thresholds for RAM >= 12 GB
        large = ram >= 12288
        g1_new = 40 if large else 30
        g1_max_new = 50 if large else 40
        g1_region = '16M' if large else '8M'
        g1_reserve = 15 if large else 20

        flags = (
            f'-Xms{ram}M -Xmx{ram}M '
            f'-XX:+UseG1GC -XX:+ParallelRefProcEnabled -XX:MaxGCPauseMillis=200 '
            f'-XX:+UnlockExperimentalVMOptions -XX:+DisableExplicitGC -XX:+AlwaysPreTouch '
            f'-XX:G1NewSizePercent={g1_new} -XX:G1MaxNewSizePercent={g1_max_new} '
            f'-XX:G1HeapRegionSize={g1_region} -XX:G1ReservePercent={g1_reserve} '
            f'-XX:G1HeapWastePercent=5 -XX:G1MixedGCCountTarget=4 '
            f'-XX:InitiatingHeapOccupancyPercent=15 -XX:G1MixedGCLiveThresholdPercent=90 '
            f'-XX:G1RSetUpdatingPauseTimePercent=5 -XX:SurvivorRatio=32 '
            f'-XX:+PerfDisableSharedMem -XX:MaxTenuringThreshold=1 '
            f'-Dusing.aikars.flags=https://mcflags.emc.gs -Daikars.new.flags=true'
        )

        ram_label = f'{ram} MB ({ram / 1024:.1f} GB)' if ram >= 1024 else f'{ram} MB'
        embed = discord.Embed(
            title='⚙️ Flags JVM de Aikar',
            color=discord.Color.dark_green(),
        )
        embed.add_field(
            name=f'RAM: {ram_label}',
            value=f'```\n{flags}\n```',
            inline=False,
        )
        embed.add_field(
            name='Como usar',
            value='Adicione essas flags ao script de inicialização, antes do `-jar`.',
            inline=False,
        )
        footer = (
            'Configuração RAM alta (>= 12 GB) • flags.sh.emc.gs'
            if large else
            'Baseado em flags.sh.emc.gs • Recomendado para servidores Minecraft'
        )
        embed.set_footer(text=footer)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name='sync', description='Sincronizar comandos slash com o Discord (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    async def sync_commands(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            await interaction.edit_original_response(content='🔄 Sincronizando comandos…')
        except discord.HTTPException:
            pass
        synced = await self.bot.tree.sync()
        try:
            await interaction.edit_original_response(content=None)
        except discord.HTTPException:
            pass
        await interaction.followup.send(
            f'✅ {len(synced)} comandos sincronizados.', ephemeral=True
        )

    @staticmethod
    async def _ping_api(
        session: aiohttp.ClientSession, url: str, label: str
    ) -> tuple[str, str, float]:
        """Ping an API endpoint and return (label, status_emoji, latency_ms)."""
        try:
            start = time.monotonic()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                latency = (time.monotonic() - start) * 1000
                if resp.status < 400:
                    return label, '🟢', latency
                return label, f'🟡 ({resp.status})', latency
        except Exception:
            return label, '🔴', 0.0

    @app_commands.command(name='health', description='Status e diagnóstico do bot (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    async def health(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            await interaction.edit_original_response(content='🏥 Verificando saúde do bot…')
        except discord.HTTPException:
            pass

        # Uptime
        elapsed = time.monotonic() - self._start_time
        days, remainder = divmod(int(elapsed), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_parts = []
        if days:
            uptime_parts.append(f'{days}d')
        if hours:
            uptime_parts.append(f'{hours}h')
        if minutes:
            uptime_parts.append(f'{minutes}m')
        uptime_parts.append(f'{seconds}s')
        uptime_str = ' '.join(uptime_parts)

        # API health checks (run concurrently)
        async with aiohttp.ClientSession() as session:
            checks = await self._check_apis(session)

        # Web search status
        if TAVILY_AVAILABLE:
            tavily_status = '🟢 Ativa'
        elif not WEB_SEARCH_ENABLED:
            tavily_status = '🔴 Desativada'
        else:
            tavily_status = '🟡 Sem API key'

        # Vector store stats
        docs_rag = self.services.docs
        chunk_count = 0
        reindex_info = 'N/A'
        source_info = ''
        if docs_rag is not None:
            chunk_count = docs_rag.chunk_count
            if docs_rag.last_commit_sha:
                reindex_info = f'`{docs_rag.last_commit_sha[:12]}`'
            if docs_rag.indexing:
                reindex_info = '⏳ Indexando...'
            # Per-source last index timestamps
            if docs_rag.source_last_index:
                lines = []
                for label, ts in docs_rag.source_last_index.items():
                    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                    lines.append(f'{label}: {dt.strftime("%Y-%m-%d %H:%M UTC")}')
                source_info = '\n'.join(lines)

        # Conversation cache stats
        conv_count = await self.services.conversations.count_active()
        conv_max = self.services.conversations.max_stored

        # Build embed
        embed = discord.Embed(
            title='🏥 Health — QuillBot',
            color=discord.Color.green(),
            description=f'Uptime: **{uptime_str}**',
        )

        # API status field
        api_lines = [f'{emoji} **{label}**' + (f' ({lat:.0f}ms)' if lat > 0 else '')
                     for label, emoji, lat in checks]
        embed.add_field(
            name='🔌 APIs',
            value='\n'.join(api_lines) or 'Nenhuma verificada',
            inline=False,
        )

        # Vector store field
        doc_lines = [f'Chunks: **{chunk_count}**']
        if reindex_info != 'N/A':
            doc_lines.append(f'Último reindex: {reindex_info}')
        if source_info:
            doc_lines.append(f'Fontes:\n{source_info}')
        embed.add_field(
            name='📚 Documentação',
            value='\n'.join(doc_lines),
            inline=True,
        )

        # Web search field
        embed.add_field(
            name='🌐 Busca Web',
            value=tavily_status,
            inline=True,
        )

        # Cache field
        embed.add_field(
            name='💬 Conversas',
            value=f'{conv_count}/{conv_max} ativas',
            inline=True,
        )

        embed.set_footer(text=f'Latência do WebSocket: {self.bot.latency * 1000:.0f}ms')
        try:
            await interaction.edit_original_response(content=None)
        except discord.HTTPException:
            pass
        await interaction.followup.send(embed=embed, ephemeral=True)

    async def _check_apis(
        self, session: aiohttp.ClientSession
    ) -> list[tuple[str, str, float]]:
        """Run API health checks concurrently and return results."""
        endpoints = [
            ('OpenRouter', 'https://openrouter.ai/api/v1/models'),
            ('Tavily', 'https://api.tavily.com'),
            ('GitHub', 'https://api.github.com'),
            ('mclo.gs', 'https://api.mclo.gs'),
            ('Modrinth', 'https://api.modrinth.com/v2/search?limit=1'),
        ]
        results = await asyncio.gather(
            *(self._ping_api(session, url, label) for label, url in endpoints)
        )
        return list(results)


async def setup(bot: commands.Bot):
    await bot.add_cog(Commands(bot))
