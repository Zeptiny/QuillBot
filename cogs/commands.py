import logging

import discord
from cachetools import TTLCache
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError

from cogs.utils import PaginatedEmbedView, split_response
from config import CHAT_MODEL, COOLDOWN_PER, COOLDOWN_RATE, DOCS_BASE_URL, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

GENERAL_SYSTEM_PROMPT = (
    "<role>\n"
    "Você é um assistente de propósito geral do servidor Miners' Refuge. "
    "Responda sempre em português brasileiro.\n"
    "</role>\n\n"
    "<instructions>\n"
    "1. Responda perguntas gerais com base no seu conhecimento.\n"
    "2. Seja honesto quando não souber a resposta — não invente informações.\n"
    "3. Quando útil, termine com uma sugestão de acompanhamento na linha final, prefixada com '💡 '.\n"
    "</instructions>\n\n"
    "<response_format>\n"
    "Seja claro e conciso. Use markdown para formatação quando aplicável.\n"
    "</response_format>"
)


class Commands(commands.Cog):
    """Slash commands for server administration help."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.client: AsyncOpenAI | None = None
        if OPENROUTER_API_KEY:
            self.client = AsyncOpenAI(
                base_url='https://openrouter.ai/api/v1',
                api_key=OPENROUTER_API_KEY,
            )
        # TTL cache: max 200 conversations, each expires after 30 min
        self._conversations: TTLCache = TTLCache(maxsize=200, ttl=1800)

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
                results = await docs_rag.search(query, top_k=5)
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
        synced = await self.bot.tree.sync()
        await interaction.followup.send(
            f'✅ {len(synced)} comandos sincronizados.', ephemeral=True
        )

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
            answer, embeds = await self._run_chat(question, image_url=image_url)
            if len(embeds) == 1:
                msg = await interaction.followup.send(embed=embeds[0], wait=True)
            else:
                msg = await interaction.followup.send(
                    embed=embeds[0], view=PaginatedEmbedView(embeds), wait=True
                )
            self._conversations[msg.id] = {
                'question': question,
                'answer': answer,
                'history': [],
            }

        except RateLimitError:
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
        except Exception:
            logger.exception("Error in /chat command")
            await interaction.followup.send(
                'Ocorreu um erro ao processar sua pergunta. Tente novamente mais tarde.'
            )

    async def _run_chat(
        self,
        question: str,
        history: list[dict] | None = None,
        image_url: str | None = None,
    ) -> tuple[str, list[discord.Embed]]:
        messages = [{'role': 'system', 'content': GENERAL_SYSTEM_PROMPT}]

        if history:
            for h in history[-3:]:
                messages.append({'role': 'user', 'content': h['question']})
                messages.append({'role': 'assistant', 'content': h['answer']})

        if image_url:
            messages.append({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question},
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                ],
            })
        else:
            messages.append({'role': 'user', 'content': question})

        response = await self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or 'Não foi possível gerar uma resposta.'

        pages = split_response(answer)
        total = len(pages)
        footer_base = "💬 Assistente geral • Miners' Refuge"
        embeds: list[discord.Embed] = []
        for i, page_text in enumerate(pages):
            e = discord.Embed(
                title=f'💬 {question}' if i == 0 else '',
                description=page_text,
                color=discord.Color.teal(),
            )
            e.set_footer(
                text=f"Página {i + 1}/{total} • {footer_base}" if total > 1 else footer_base
            )
            embeds.append(e)

        return answer, embeds

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
        """Handle reply-based follow-up conversations for /chat."""
        if message.author.bot:
            return
        if not message.reference or not message.reference.message_id:
            return

        ref_id = message.reference.message_id
        conv = self._conversations.get(ref_id)
        if not conv:
            return

        follow_up_question = message.content.strip()
        if not follow_up_question and not message.attachments:
            return

        image_url = None
        for att in message.attachments:
            if att.content_type and att.content_type.startswith('image/'):
                image_url = att.url
                break

        if not follow_up_question:
            follow_up_question = 'Analise esta imagem.'

        async with message.channel.typing():
            try:
                history = conv.get('history', []).copy()
                history.append({'question': conv['question'], 'answer': conv['answer']})

                answer, embeds = await self._run_chat(
                    follow_up_question, history=history, image_url=image_url
                )

                if len(embeds) == 1:
                    reply = await message.reply(embed=embeds[0])
                else:
                    reply = await message.reply(
                        embed=embeds[0], view=PaginatedEmbedView(embeds)
                    )

                self._conversations[reply.id] = {
                    'question': follow_up_question,
                    'answer': answer,
                    'history': history,
                }
            except RateLimitError:
                await message.reply(
                    '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
                )
            except Exception:
                logger.exception("Error in /chat follow-up reply")
                await message.reply(
                    'Ocorreu um erro ao processar sua pergunta. Tente novamente.'
                )


async def setup(bot: commands.Bot):
    await bot.add_cog(Commands(bot))
