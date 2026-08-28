import logging
import datetime
import re
import time

import aiohttp
import discord
from cachetools import TTLCache
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError

from cogs.conversation_store import (
    ConversationStore,
    add_participant,
    author_info,
    build_conversation_block,
    build_current_message,
    build_history_messages,
    cap_turns,
    make_turn,
)
from cogs.memory import MEMORY_ABOUT_TOOL, MEMORY_SEARCH_TOOL, MEMORY_WRITE_TOOL
from cogs.tavily_tools import TOOLS as TAVILY_TOOLS
from cogs.tavily_tools import exec_tool as tavily_exec_tool
from cogs.tavily_tools import status_label as tavily_status_label
from cogs.tavily_tools import MAX_TOOL_ROUNDS as TAVILY_MAX_ROUNDS
from cogs.utils import (
    CHANNEL_HISTORY_TOOL,
    COUNT_MENTIONS_TOOL,
    GET_MESSAGE_CONTEXT_TOOL,
    GET_USER_STATS_TOOL,
    GUILD_INFO_TOOL,
    PaginatedEmbedView,
    SEARCH_HISTORY_TOOL,
    build_full_context_block,
    build_guild_context,
    build_temporal_context,
    build_user_context,
    fetch_channel_history,
    fetch_message_context,
    build_source_pages,
    run_tool_loop,
    split_response,
)
from config import (
    CHAT_MENTION_ENABLED,
    CHAT_MODEL,
    CONVERSATIONS_DB_PATH,
    CONVERSATIONS_HISTORY_TURNS,
    CONVERSATIONS_MAX_STORED,
    CONVERSATIONS_MAX_TURNS,
    CONVERSATIONS_TTL_SECONDS,
    COOLDOWN_PER,
    COOLDOWN_RATE,
    DOCS_BASE_URL,
    MEMORY_ENABLED,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    TAVILY_AVAILABLE,
)

logger = logging.getLogger(__name__)


def _conversation_participants(message: discord.Message) -> set[str]:
    """User ids relevant to a message: author, mentions and reply target."""
    ids = {str(message.author.id)}
    for u in getattr(message, 'mentions', []):
        if not u.bot:
            ids.add(str(u.id))
    ref = getattr(message, 'reference', None)
    ref_msg = getattr(ref, 'resolved', None)
    ref_author = getattr(ref_msg, 'author', None)
    if ref_author is not None and not ref_author.bot:
        ids.add(str(ref_author.id))
    return ids

_MEMORY_INSTRUCTIONS = (
    "- Você tem uma memória persistente: um bloco <memory> com lembranças "
    "relevantes é injetado automaticamente no início de cada resposta. Use-o "
    "naturalmente — nunca mencione o bloco nem os IDs [mem #N].\n"
    "- Para lembrar de algo não injetado, use `memory_search`; para contextualizar "
    "quem é alguém, use `memory_about`.\n"
    "- Ao descobrir algo digno de memória (preferência estável, fato sobre pessoa, "
    "evento marcante, habilidade ensinada), salve com `memory_write` — uma frase "
    "autocontida por memória, em terceira pessoa. Atualize com action=update quando "
    "o fato mudar e use action=forget quando deixar de valer. Use pinned=true só "
    "para fatos centrais e permanentes.\n"
) if MEMORY_ENABLED else ""

_WEB_SEARCH_INSTRUCTIONS = (
    "- Para informações em tempo real ou recentes, use as ferramentas de busca web.\n"
    "  - Use `web_search` para buscar informações atualizadas na web.\n"
    "  - Use `web_extract` quando precisar ler o conteúdo completo de uma URL específica.\n"
    "  - Prefira `web_search` primeiro; use `web_extract` para se aprofundar em fontes relevantes.\n"
    "  - Use `search_depth='advanced'` para buscas mais precisas quando necessário.\n"
    "- Quando citar resultados da web, inclua o título e o link da fonte.\n"
)

_DISCORD_FORMAT = (
    "A resposta será exibida no Discord (embed description) — use APENAS sintaxe que o Discord renderiza:\n"
    "- Permitido: **negrito**, *itálico*, __sublinhado__, ~~tachado~~, `código inline`, "
    "```bloco de código``` com linguagem (yaml, properties, json, toml, bash, log), "
    "> citação, - lista, 1. lista numerada, ||spoiler||, ### título / ## título, [texto](url).\n"
    "- Proibido: tabelas markdown com |, separadores --- ou ***, HTML (<br>, <div>), LaTeX ($$), footnotes.\n"
    "- Para comparações use listas com **Chave**: valor — NUNCA tabelas com pipes.\n"
    "- Para títulos de seção use ### Título ou **Negrito** — nunca ---.\n"
    "- Blocos de código sempre com linguagem; valores de config em ```yaml ou ```properties.\n"
)

GENERAL_SYSTEM_PROMPT = (
    "<role>\n"
    "Você é um assistente de propósito geral do servidor Miners' Refuge. "
    "Responda sempre em português brasileiro.\n"
    "</role>\n\n"
    "<instructions>\n"
    "- Responda perguntas gerais com base no seu conhecimento.\n"
    + _MEMORY_INSTRUCTIONS
    + (_WEB_SEARCH_INSTRUCTIONS if TAVILY_AVAILABLE else '') +
    "- Seja honesto quando não souber a resposta — não invente informações.\n"
    "</instructions>\n\n"
    "<response_format>\n"
    + _DISCORD_FORMAT +
    "</response_format>"
)


class Commands(commands.Cog):
    """Slash commands for server administration help."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.client: AsyncOpenAI | None = None
        api_key = OPENAI_API_KEY or "not-needed"
        try:
            self.client = AsyncOpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=api_key,
            )
            if not OPENAI_API_KEY and 'openrouter.ai' in OPENAI_BASE_URL:
                logger.warning("OPENAI_API_KEY not set -- /chat will fail until configured")
        except Exception:
            logger.exception("Failed to initialize OpenAI client for Commands cog")
            self.client = None
        # Persistent conversations with activity-based TTL (survive restarts)
        self.store = ConversationStore(
            CONVERSATIONS_DB_PATH,
            kind='chat',
            ttl_seconds=CONVERSATIONS_TTL_SECONDS,
            max_stored=CONVERSATIONS_MAX_STORED,
        )
        # Per-user follow-up cooldown (same period as slash commands)
        self._followup_cd: TTLCache = TTLCache(maxsize=500, ttl=COOLDOWN_PER)
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
            docs_rag = self.bot.cogs.get('DocsRAG')
            if docs_rag and hasattr(docs_rag, '_indexing') and docs_rag._indexing:
                await interaction.response.send_message(
                    '📚 A documentação está sendo indexada. Tente novamente em alguns instantes.',
                    ephemeral=True,
                )
                return
            if docs_rag and hasattr(docs_rag, 'search') and docs_rag.chunks:
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
                    from cogs.docs_rag import path_to_docs_url
                    seen_paths = set()
                    lines = []
                    for r in results:
                        if r['path'] not in seen_paths:
                            seen_paths.add(r['path'])
                            url = r.get('doc_url', path_to_docs_url(r['path']))
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
        for cmd in sorted(cmds, key=lambda c: c.name):
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
        tavily_status = '🟢 Ativa' if TAVILY_AVAILABLE else ('🔴 Desativada' if not TAVILY_API_KEY else '🟡 Sem API key')

        # Vector store stats
        docs_rag = self.bot.cogs.get('DocsRAG')
        chunk_count = 0
        reindex_info = 'N/A'
        source_info = ''
        if docs_rag:
            chunk_count = len(docs_rag.chunks)
            if docs_rag._last_commit_sha:
                reindex_info = f'`{docs_rag._last_commit_sha[:12]}`'
            if docs_rag._indexing:
                reindex_info = '⏳ Indexando...'
            # Per-source last index timestamps
            if hasattr(docs_rag, '_source_last_index') and docs_rag._source_last_index:
                import datetime
                lines = []
                for label, ts in docs_rag._source_last_index.items():
                    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                    lines.append(f'{label}: {dt.strftime("%Y-%m-%d %H:%M UTC")}')
                source_info = '\n'.join(lines)

        # Conversation cache stats
        conv_count = await self.store.count_active()
        conv_max = self.store.max_stored

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
        import asyncio
        results = await asyncio.gather(
            *(self._ping_api(session, url, label) for label, url in endpoints)
        )
        return list(results)

    @app_commands.command(name='chat', description='Faça uma pergunta geral ao assistente')
    @app_commands.checks.cooldown(COOLDOWN_RATE, COOLDOWN_PER)
    @app_commands.describe(
        question='Sua pergunta',
        image='Imagem/screenshot para análise (opcional)',
    )
    async def chat(
        self,
        interaction: discord.Interaction,
        question: str,
        image: discord.Attachment | None = None,
    ):
        if not self.client:
            await interaction.response.send_message(
                '⚠️ Comando indisponível: chave de API não configurada.', ephemeral=True
            )
            return

        image_url = None
        if image:
            if not image.content_type or not image.content_type.startswith('image/'):
                await interaction.response.send_message(
                    'O arquivo enviado não é uma imagem válida.', ephemeral=True
                )
                return
            image_url = image.url

        await interaction.response.defer(thinking=True)
        try:
            await interaction.edit_original_response(content='💭 Pensando…')
        except discord.HTTPException:
            pass

        logger.info(
            "Processing /chat user=%s guild=%s question=%r",
            interaction.user.id, interaction.guild_id, question[:80],
        )

        try:
            answer, embeds, sources = await self._run_chat(
                question,
                image_url=image_url,
                interaction=interaction,
                user=interaction.user,
                guild=interaction.guild,
                channel=interaction.channel,
                created_at=interaction.created_at,
            )
            # Clear any in-progress status message before sending the final embed.
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            if len(embeds) == 1:
                msg = await interaction.followup.send(embed=embeds[0], wait=True)
            else:
                msg = await interaction.followup.send(
                    embed=embeds[0], view=PaginatedEmbedView(embeds), wait=True
                )
            await self._store_new_conversation(
                msg, question, answer, sources,
                user=interaction.user,
                guild=interaction.guild,
                channel=interaction.channel,
                created_at=interaction.created_at,
                images=[image_url] if image_url else [],
            )

        except RateLimitError:
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
        except Exception:
            logger.exception("Error in /chat command")
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send(
                'Ocorreu um erro ao processar sua pergunta. Tente novamente mais tarde.'
            )

    async def _store_new_conversation(
        self,
        reply_msg: discord.Message,
        question: str,
        answer: str,
        sources: list[dict],
        *,
        user,
        guild,
        channel,
        created_at: datetime.datetime | None,
        images: list[str] | None = None,
        message_id: str | int | None = None,
        reply_to: str | int | None = None,
    ) -> None:
        """Persist a brand-new conversation anchored on the bot's reply message."""
        author = author_info(user)
        ts = created_at.timestamp() if created_at else time.time()
        turn = make_turn(
            question, answer,
            author=author, ts=ts,
            message_id=message_id,
            channel_id=getattr(channel, 'id', None),
            channel_name=getattr(channel, 'name', None),
            images=images,
            sources=sources,
            reply_to=reply_to,
        )
        origin = {
            'channel_id': str(getattr(channel, 'id', '') or ''),
            'channel_name': getattr(channel, 'name', '') or '',
            'guild_id': str(getattr(guild, 'id', '') or ''),
            'guild_name': getattr(guild, 'name', '') or '',
        }
        await self.store.create(
            str(reply_msg.id),
            guild_id=origin['guild_id'] or None,
            channel_id=origin['channel_id'] or None,
            data={
                'turns': [turn],
                'participants': [author] if author.get('id') else [],
                'origin': origin,
                'started_ts': ts,
            },
        )

    async def _run_chat(
        self,
        question: str,
        history: list[dict] | None = None,
        image_url: str | None = None,
        image_urls: list[str] | None = None,
        interaction: discord.Interaction | None = None,
        user: discord.abc.User | discord.Member | None = None,
        guild: discord.Guild | None = None,
        channel: discord.abc.GuildChannel | discord.Thread | discord.DMChannel | None = None,
        created_at: datetime.datetime | None = None,
        reply_to: str | None = None,
        participant_ids: set[str] | None = None,
        origin: str | None = None,
    ) -> tuple[str, list[discord.Embed], list[dict]]:
        context_block = None
        if user or guild or channel:
            try:
                context_block = build_full_context_block(user or (interaction.user if interaction else None), guild or (interaction.guild if interaction else None), channel or (interaction.channel if interaction else None), created_at or (interaction.created_at if interaction else None))
            except Exception:
                logger.exception("Failed to build context block")
        system_content = GENERAL_SYSTEM_PROMPT
        if context_block:
            system_content = f"{GENERAL_SYSTEM_PROMPT}\n\n{context_block}"
        history = history or []
        if history:
            replay = history[-CONVERSATIONS_HISTORY_TURNS:]
            system_content += '\n\n' + build_conversation_block(
                replay, current_author=author_info(user or (interaction.user if interaction else None))
            )
        if MEMORY_ENABLED:
            mem_cog = self.bot.get_cog('Memory')
            if mem_cog is not None and (guild or (interaction.guild if interaction else None)):
                try:
                    mem_block = await mem_cog.build_memory_block(
                        (guild or interaction.guild).id, question,
                        speaker_id=str(user.id) if user is not None else None,
                        participant_ids=participant_ids,
                    )
                    if mem_block:
                        system_content += f'\n\n{mem_block}'
                except Exception:
                    logger.exception('Failed to build memory block')
        messages: list[dict] = [{'role': 'system', 'content': system_content}]

        if history:
            messages.extend(
                build_history_messages(history, max_turns=CONVERSATIONS_HISTORY_TURNS)
            )

        urls: list[str] = []
        if image_urls:
            urls.extend([u for u in image_urls if u])
        elif image_url:
            urls.append(image_url)
        messages.append(build_current_message(
            question,
            author=author_info(user or (interaction.user if interaction else None)),
            ts=created_at.timestamp() if created_at else None,
            image_urls=urls,
            reply_to=reply_to,
            in_conversation=bool(history),
        ))

        base_tools = list(TAVILY_TOOLS) if TAVILY_AVAILABLE else []
        base_tools.extend([CHANNEL_HISTORY_TOOL, GUILD_INFO_TOOL, SEARCH_HISTORY_TOOL, GET_MESSAGE_CONTEXT_TOOL, GET_USER_STATS_TOOL, COUNT_MENTIONS_TOOL])
        if MEMORY_ENABLED:
            base_tools.extend([MEMORY_SEARCH_TOOL, MEMORY_WRITE_TOOL, MEMORY_ABOUT_TOOL])
        active_tools = base_tools if base_tools else None
        fallback_channel = channel or (interaction.channel if interaction else None)
        fallback_guild = guild or (interaction.guild if interaction else None)

        async def _exec(name: str, args: dict) -> tuple[str, list[dict]]:
            if name in ('web_search', 'web_extract'):
                return await tavily_exec_tool(name, args)
            if name in ('memory_search', 'memory_write', 'memory_about'):
                mem_cog = self.bot.get_cog('Memory')
                if not mem_cog:
                    return 'Memória não disponível.', []
                g = fallback_guild
                if not g:
                    return 'Memória requer estar em um servidor.', []
                actor_name = f'bot (via {user.display_name})' if user is not None else 'bot'
                return await mem_cog.exec_tool(
                    name, args, guild=g, actor_name=actor_name,
                    requester=user, channel=channel, origin=origin,
                    participant_ids=participant_ids,
                )
            if name == 'get_channel_history':
                text = await fetch_channel_history(self.bot, fallback_channel, limit=args.get('limit', 20), channel_id=args.get('channel_id'))
                return text, []
            if name == 'get_guild_info':
                g = guild or (interaction.guild if interaction else None)
                if not g:
                    return "Fora de um servidor (DM).", []
                text = build_guild_context(g)
                # add channel list
                try:
                    chs = [f"#{c.name} ({c.id})" for c in g.channels if isinstance(c, discord.TextChannel)][:30]
                    if chs:
                        text += "\nCanais de texto: " + ", ".join(chs)
                except Exception:
                    pass
                return text, []
            if name == 'search_history':
                hist = self.bot.get_cog('HistoryRAG')
                if not hist:
                    return "Histórico não disponível.", []
                g = fallback_guild
                if not g:
                    return "Busca no histórico requer estar em um servidor.", []
                query = args.get('query', '')
                limit = max(1, min(12, int(args.get('limit', 5))))
                try:
                    results = await hist.search(query, g.id, limit=limit, channel_id=args.get('channel_id'), author_id=args.get('author_id'), author_name=args.get('author_name'), after=args.get('after'), before=args.get('before'), search_mode=args.get('search_mode','hybrid'), sort_by=args.get('sort_by','relevance'))  # type: ignore
                except Exception:
                    logger.exception("search_history failed")
                    return "Erro ao buscar no histórico.", []
                if not results:
                    return "Nenhuma mensagem relevante encontrada no histórico.", []
                parts = []
                for r in results:
                    jump = r.get('jump_url', '')
                    link = f"[ver]({jump})" if jump else ""
                    window = r.get('chunk_text', r.get('content', ''))[:1200]
                    header = f"**{r.get('author_full','?')}** em #{r.get('channel_name','?')} — {r.get('ts','')} {link} (score {r.get('_score',0):.2f})"
                    parts.append(f"{header}\n```\n{window}\n```\n`msg_id={r.get('msg_id')} channel_id={r.get('channel_id')}`")
                return "\n\n---\n\n".join(parts), []
            if name == 'get_user_stats':
                hist = self.bot.get_cog('HistoryRAG')
                if not hist:
                    return "Histórico não disponível.", []
                g = fallback_guild
                if not g:
                    return "Requer servidor.", []
                try:
                    stats = await hist.get_user_stats(g.id, author_id=args.get('author_id'), author_name=args.get('author_name'))  # type: ignore
                except Exception:
                    logger.exception("get_user_stats failed")
                    return "Erro ao buscar estatísticas.", []
                if "error" in stats:
                    return stats["error"], []
                lines = [f"Usuário: {stats['author_full']} ({', '.join(stats['author_ids'])})", f"Total: {stats['total_messages']} msgs | Média: {stats['avg_length']} chars", f"Canais: {', '.join(f'{k}={v}' for k,v in stats['top_channels'])}", f"Horários: {', '.join(f'{h}h={v}' for h,v in stats['top_hours'])}", f"Período: {stats['first_seen']} → {stats['last_seen']}", f"Exemplo: {stats['example_content']} {stats['example_jump']}"]
                return "\n".join(lines), []
            if name == 'count_mentions':
                hist = self.bot.get_cog('HistoryRAG')
                if not hist:
                    return "Histórico não disponível.", []
                g = fallback_guild
                if not g:
                    return "Requer servidor.", []
                try:
                    groups = await hist.count_mentions(g.id, query=args.get('query',''), group_by=args.get('group_by','author'), limit=int(args.get('limit',10)), after=args.get('after'), before=args.get('before'))  # type: ignore
                except Exception:
                    logger.exception("count_mentions failed")
                    return "Erro ao contar menções.", []
                if not groups:
                    return "Nenhuma menção encontrada.", []
                lines = [f"{gr['key']}: {gr['count']}× — ex: {gr['example'].get('content','')[:120]}" for gr in groups]
                return "\n".join(lines), []
            if name == 'get_message_context':
                text = await fetch_message_context(self.bot, channel_id=args.get('channel_id',''), message_id=args.get('message_id',''), window=args.get('window', 5))
                return text, []
            return f'Ferramenta desconhecida: {name}', []

        def _status(name: str, args: dict) -> str:
            if name in ('web_search', 'web_extract'):
                return tavily_status_label(name, args)
            if name == 'memory_search':
                return f"🧠 Recordando: *{args.get('query', '')[:40]}*"
            if name == 'memory_write':
                act = args.get('action', 'write')
                tgt = str(args.get('memory_id') or args.get('content') or args.get('content_match') or '')[:40]
                return f'🧠 Memória — {act}: *{tgt}*'
            if name == 'memory_about':
                return f"🧠 Relembrando {args.get('user', 'quem pergunta')}…"
            if name == 'get_channel_history':
                lim = args.get('limit', 20)
                cid = args.get('channel_id')
                return f'📜 Lendo histórico ({lim} msgs)' + (f' canal {cid}' if cid else '')
            if name == 'get_guild_info':
                return '🏰 Coletando informações do servidor'
            if name == 'search_history':
                q = args.get('query','')[:40]
                return f'🔎 Buscando no histórico: *{q}*'
            if name == 'get_user_stats':
                return f'📊 Estatísticas de {args.get("author_id") or args.get("author_name","usuário")}…'
            if name == 'count_mentions':
                return f'🔢 Contando menções: *{args.get("query","")[:30]}*'
            if name == 'get_message_context':
                return f'🧩 Contexto da mensagem {args.get("message_id","")}…'
            return f'🔧 Executando: {name}'

        answer, all_sources = await run_tool_loop(
            client=self.client,
            model=CHAT_MODEL,
            messages=messages,
            tools=active_tools,
            exec_tool=_exec,
            status_label=_status,
            interaction=interaction,
            max_rounds=TAVILY_MAX_ROUNDS,
            dedup_key=lambda s: s.get('url', ''),
        )

        source_lines: list[str] = []
        sources: list[dict] = []
        if all_sources:
            seen_src: set[str] = set()
            for s in all_sources:
                url = s.get('url', '')
                if not url or url in seen_src:
                    continue
                seen_src.add(url)
                title = (s.get('title') or url)[:100]
                source_lines.append(f'• [{title}]({url})')
                sources.append({'title': title, 'url': url})

        pages = split_response(answer)
        total = len(pages)
        embeds: list[discord.Embed] = []
        for i, page_text in enumerate(pages):
            e = discord.Embed(
                description=page_text,
                color=discord.Color.teal(),
            )
            page_num = i + 1
            if total > 1:
                e.set_footer(text=f"Página {page_num}/{total}")
            embeds.append(e)

        if source_lines:
            embeds.extend(
                build_source_pages(
                    source_lines,
                    title='🌐 Fontes da Web',
                    color=discord.Color.teal(),
                )
            )

        return answer, embeds, sources

    async def cog_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ):
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(
                f'⏳ Aguarde {error.retry_after:.0f}s antes de usar este comando novamente.',
                ephemeral=True,
            )
        else:
            raise error

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle reply-based follow-up conversations and @mention chat mode."""
        if message.author.bot:
            return
        # --- 1) Reply-based follow-up (existing conversation) ---
        if message.reference and message.reference.message_id:
            ref_id = message.reference.message_id
            conv = await self.store.get_by_handle(ref_id)
            if conv:
                user_id = message.author.id
                if user_id in self._followup_cd:
                    try:
                        await message.reply('⏳ Aguarde antes de enviar outra resposta.', delete_after=5)
                    except discord.HTTPException:
                        pass
                    return
                self._followup_cd[user_id] = True
                follow_up_question = message.content.strip()
                if self.bot.user:
                    follow_up_question = re.sub(rf'<@!?{self.bot.user.id}>', '', follow_up_question)
                    follow_up_question = re.sub(r'\s+', ' ', follow_up_question).strip()
                if not follow_up_question and not message.attachments:
                    return
                image_urls = [att.url for att in message.attachments if att.content_type and att.content_type.startswith('image/')]
                if not follow_up_question:
                    follow_up_question = 'Analise esta imagem.'
                async with message.channel.typing():
                    try:
                        history = conv['data'].get('turns', []).copy()
                        answer, embeds, sources = await self._run_chat(
                            follow_up_question,
                            history=history,
                            image_urls=image_urls if image_urls else None,
                            user=message.author,
                            guild=message.guild,
                            channel=message.channel,
                            created_at=message.created_at,
                            reply_to=str(ref_id),
                            participant_ids=_conversation_participants(message),
                            origin=message.jump_url,
                        )
                        if len(embeds) == 1:
                            reply = await message.reply(embed=embeds[0])
                        else:
                            reply = await message.reply(
                                embed=embeds[0], view=PaginatedEmbedView(embeds)
                            )
                        turn = make_turn(
                            follow_up_question, answer,
                            author=author_info(message.author),
                            ts=message.created_at.timestamp(),
                            message_id=message.id,
                            channel_id=message.channel.id,
                            channel_name=getattr(message.channel, 'name', None),
                            images=image_urls,
                            sources=sources,
                            reply_to=ref_id,
                        )
                        data = conv['data']
                        data['turns'] = cap_turns(history + [turn], CONVERSATIONS_MAX_TURNS)
                        add_participant(data.setdefault('participants', []), author_info(message.author))
                        await self.store.update(
                            conv['conv_id'], data, new_handle_msg_id=reply.id,
                        )
                    except RateLimitError:
                        await message.reply(
                            '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
                        )
                    except Exception:
                        logger.exception("Error in /chat follow-up reply")
                        await message.reply(
                            'Ocorreu um erro ao processar sua pergunta. Tente novamente.'
                        )
                return
        # --- 2) @mention chat mode (same as /chat) ---
        if not CHAT_MENTION_ENABLED:
            return
        if not self.bot.user or not self.bot.user.mentioned_in(message):
            return
        if not self.client:
            try:
                await message.reply('⚠️ Chat indisponível: chave de API não configurada.', mention_author=False)
            except discord.HTTPException:
                pass
            return
        # Per-user cooldown for mentions (reuse followup cooldown)
        user_id = message.author.id
        if user_id in self._followup_cd:
            try:
                await message.reply('⏳ Aguarde antes de enviar outra pergunta.', delete_after=5)
            except discord.HTTPException:
                pass
            return
        # Strip bot mention to get clean question
        mention_pattern = re.compile(rf'<@!?{self.bot.user.id}>')
        clean_question = mention_pattern.sub('', message.content).strip()
        # Also strip any extra mention artifacts and whitespace
        clean_question = re.sub(r'\s+', ' ', clean_question).strip()
        current_image_urls = [att.url for att in message.attachments if att.content_type and att.content_type.startswith('image/')]
        ref_context = ""
        ref_image_urls: list[str] = []
        if message.reference and message.reference.message_id:
            ref_msg = message.reference.resolved
            if ref_msg is None:
                try:
                    ref_msg = await message.channel.fetch_message(message.reference.message_id)
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    ref_msg = None
                except Exception:
                    logger.exception("Failed to fetch referenced message %s", message.reference.message_id)
                    ref_msg = None
            if ref_msg:
                try:
                    ref_author = getattr(ref_msg.author, 'display_name', str(ref_msg.author))
                    ref_content = (ref_msg.content or "").strip()
                    if len(ref_content) > 1500:
                        ref_content = ref_content[:1500] + "…"
                    for att in getattr(ref_msg, 'attachments', []):
                        if att.content_type and att.content_type.startswith('image/'):
                            ref_image_urls.append(att.url)
                    attach_names = [att.filename for att in getattr(ref_msg, 'attachments', []) if not (att.content_type and att.content_type.startswith('image/'))]
                    parts = [f"[Mensagem respondida — {ref_author} (@{ref_msg.author.name})]"]
                    if ref_content:
                        parts.append(f"Conteúdo: {ref_content}")
                    if attach_names:
                        parts.append(f"Anexos: {', '.join(attach_names[:5])}")
                    if ref_msg.attachments and not ref_content and not attach_names and ref_image_urls:
                        parts.append(f"Anexos: {len(ref_image_urls)} imagem(ns)")
                    if not ref_content and not ref_msg.attachments and ref_msg.embeds:
                        try:
                            embed = ref_msg.embeds[0]
                            embed_text = (embed.description or embed.title or "")[:500]
                            if embed_text:
                                parts.append(f"Embed: {embed_text}")
                        except Exception:
                            pass
                    if len(parts) > 1:
                        ref_context = "\n".join(parts)
                except Exception:
                    logger.exception("Failed to build ref context")
        all_image_urls = current_image_urls + ref_image_urls
        if ref_context:
            if clean_question:
                clean_question = f"{ref_context}\n---\n{clean_question}"
            else:
                clean_question = f"{ref_context}\n---\nAnalise a mensagem e imagem(ns) respondida(s) acima."
        if not clean_question and not all_image_urls:
            try:
                await message.reply(
                    f'Olá {message.author.mention}! Me mencione com uma pergunta. Ex: @{self.bot.user.display_name} como otimizar meu servidor?',
                    mention_author=False,
                )
            except discord.HTTPException:
                pass
            return
        if not clean_question and all_image_urls:
            clean_question = 'Analise esta imagem.'
        self._followup_cd[user_id] = True
        logger.info("Processing @mention chat user=%s guild=%s question=%r ref=%s images=%d", message.author.id, message.guild.id if message.guild else None, clean_question[:80], bool(ref_context), len(all_image_urls))
        async with message.channel.typing():
            try:
                answer, embeds, sources = await self._run_chat(
                    clean_question,
                    image_urls=all_image_urls if all_image_urls else None,
                    user=message.author,
                    guild=message.guild,
                    channel=message.channel,
                    created_at=message.created_at,
                    participant_ids=_conversation_participants(message),
                    origin=message.jump_url,
                )
                if len(embeds) == 1:
                    reply = await message.reply(embed=embeds[0], mention_author=False)
                else:
                    reply = await message.reply(embed=embeds[0], view=PaginatedEmbedView(embeds), mention_author=False)
                await self._store_new_conversation(
                    reply, clean_question, answer, sources,
                    user=message.author,
                    guild=message.guild,
                    channel=message.channel,
                    created_at=message.created_at,
                    images=all_image_urls,
                    message_id=message.id,
                )
            except RateLimitError:
                await message.reply('⏳ Limite de requisições atingido. Tente novamente em alguns minutos.', mention_author=False)
            except Exception:
                logger.exception("Error in @mention chat")
                await message.reply('Ocorreu um erro ao processar sua pergunta. Tente novamente.', mention_author=False)


async def setup(bot: commands.Bot):
    await bot.add_cog(Commands(bot))
