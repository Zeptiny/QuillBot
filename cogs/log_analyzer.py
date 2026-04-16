import logging
import re

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError

from config import (
    CHAT_MODEL,
    COOLDOWN_PER,
    COOLDOWN_RATE,
    MAX_CONTENT_SIZE,
    MAX_LOG_CONTEXT,
    OPENROUTER_API_KEY,
)
from responses.errors import patterns

logger = logging.getLogger(__name__)

MCLO_GS_PATTERN = re.compile(r'https://mclo\.gs/(\w+)')
PASTEBIN_PATTERN = re.compile(r'https://pastebin\.com/(\w+)')

ANALYZE_SYSTEM_PROMPT = (
    "Você é um especialista em administração de servidores Minecraft. "
    "Analise o log/crash report fornecido e responda em português brasileiro.\n\n"
    "Sua resposta DEVE seguir este formato:\n\n"
    "## 🔍 Resumo\n"
    "Uma frase resumindo o estado geral do servidor.\n\n"
    "## ❌ Erros Encontrados\n"
    "Liste cada erro encontrado com:\n"
    "- **Erro**: Descrição do erro\n"
    "- **Causa provável**: O que pode estar causando\n"
    "- **Solução**: Passos para resolver\n\n"
    "## ⚠️ Avisos\n"
    "Avisos que não são críticos mas merecem atenção.\n\n"
    "## 💡 Recomendações\n"
    "Sugestões de otimização ou melhorias baseadas no log.\n\n"
    "Se o log estiver limpo sem erros, diga isso claramente. "
    "Seja direto e prático. Não invente problemas que não existem no log."
)

# Minimum message length to run pattern matching against (skip trivial messages)
_MIN_PATTERN_LENGTH = 20


def _sanitize_log_content(text: str) -> str:
    """Strip non-printable characters from log content, keeping newlines and tabs."""
    return re.sub(r'[^\x09\x0a\x0d\x20-\x7e\x80-\uffff]', '', text)


def _truncate_safe(text: str, limit: int = 3800) -> str:
    """Truncate text at the last newline before *limit* to avoid breaking markdown."""
    if len(text) <= limit:
        return text
    idx = text.rfind('\n', 0, limit)
    if idx == -1:
        idx = limit
    return text[:idx] + '\n\n*...análise truncada*'


def check_message(content: str) -> str | None:
    """Check message content against known error patterns."""
    for compiled_pattern, response_template in patterns:
        match = compiled_pattern.search(content)
        if match:
            return response_template.format(*match.groups())
    return None


def _parse_link(text: str) -> tuple[str | None, str | None]:
    """Parse a paste-service URL and return (fetch_url, display_url).

    For mclo.gs links the fetch URL points to the raw API endpoint.
    """
    mclogs_match = MCLO_GS_PATTERN.search(text)
    if mclogs_match:
        paste_id = mclogs_match.group(1)
        return f'https://api.mclo.gs/1/raw/{paste_id}', mclogs_match.group(0)
    pastebin_match = PASTEBIN_PATTERN.search(text)
    if pastebin_match:
        return f'https://pastebin.com/raw/{pastebin_match.group(1)}', None
    return None, None


class LogAnalyzer(commands.Cog):
    """Analyzes logs and error messages sent in chat."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.session: aiohttp.ClientSession | None = None
        self.ai_client: AsyncOpenAI | None = None
        if OPENROUTER_API_KEY:
            self.ai_client = AsyncOpenAI(
                base_url='https://openrouter.ai/api/v1',
                api_key=OPENROUTER_API_KEY,
            )

    async def cog_load(self):
        self.session = aiohttp.ClientSession()

    async def cog_unload(self):
        if self.session:
            await self.session.close()

    # --- Cooldown error handler ---

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

    # --- HTTP helpers ---

    async def read_file_content(self, url: str) -> str | None:
        """Fetch text content from a URL, streaming with a hard size limit."""
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                # Reject early if Content-Length is present and too large
                content_length = resp.headers.get('Content-Length')
                if content_length and int(content_length) > MAX_CONTENT_SIZE:
                    logger.warning("Content too large from %s (%s bytes)", url, content_length)
                    return None
                # Stream in chunks to enforce the limit regardless of headers
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > MAX_CONTENT_SIZE:
                        logger.warning("Streaming content exceeded size limit from %s", url)
                        return None
                    chunks.append(chunk)
                return b''.join(chunks).decode('utf-8', errors='replace')
        except Exception:
            logger.exception("Failed to fetch content from %s", url)
            return None

    async def upload_mclogs(self, content: str) -> str | None:
        """Upload log content to mclo.gs and return the URL."""
        try:
            async with self.session.post(
                'https://api.mclo.gs/1/log',
                data={'content': content}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('url')
        except Exception:
            logger.exception("Failed to upload to mclo.gs")
        return None

    # --- Passive message listener ---

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        has_attachments = bool(message.attachments)
        content = message.content

        # Quick pre-filter: skip short messages without attachments or URLs
        if len(content) < _MIN_PATTERN_LENGTH and not has_attachments:
            if 'mclo.gs/' not in content and 'pastebin.com/' not in content:
                return

        response = check_message(content)

        # Check for paste service links
        fetch_url, mclogs_url = _parse_link(content)

        if fetch_url:
            await message.add_reaction('👀')
            link_content = await self.read_file_content(fetch_url)
            if link_content and not response:
                response = check_message(link_content)

        # Check file attachments
        if has_attachments:
            for attachment in message.attachments:
                if attachment.filename.endswith(('.txt', '.log')):
                    file_content = await self.read_file_content(attachment.url)
                    if file_content:
                        if not response:
                            response = check_message(file_content)
                        mclogs_url = await self.upload_mclogs(file_content)
                        if mclogs_url:
                            await message.reply(
                                f'Na próxima vez busque utilizar um serviço para enviar suas logs, '
                                f'como o mclo.gs, fiz o upload para você <3:\n<{mclogs_url}>'
                            )
                        else:
                            await message.reply(
                                'Não consegui fazer o upload para o mclo.gs. '
                                'Você pode enviar manualmente em https://mclo.gs'
                            )
                        if response:
                            break

        if response:
            await message.reply(response)

    # --- AI Log Analysis ---

    async def _analyze_with_ai(self, log_content: str | None = None, image_url: str | None = None) -> str | None:
        """Send log content and/or image to the LLM for analysis."""
        if not self.ai_client:
            return None

        messages = [{'role': 'system', 'content': ANALYZE_SYSTEM_PROMPT}]

        user_parts = []

        if log_content:
            log_content = _sanitize_log_content(log_content)
            # Truncate if too long, keeping start and end (most useful parts)
            if len(log_content) > MAX_LOG_CONTEXT:
                half = MAX_LOG_CONTEXT // 2
                log_content = (
                    log_content[:half]
                    + '\n\n[... log truncado ...]\n\n'
                    + log_content[-half:]
                )
            user_parts.append({'type': 'text', 'text': f"Analise este log:\n\n```\n{log_content}\n```"})

        if image_url:
            if not user_parts:
                user_parts.append({'type': 'text', 'text': 'Analise esta imagem de log/erro de servidor Minecraft:'})
            user_parts.append({'type': 'image_url', 'image_url': {'url': image_url}})

        if not user_parts:
            return None

        # Use vision format (list of content parts) when image is present
        if image_url:
            messages.append({'role': 'user', 'content': user_parts})
        else:
            messages.append({'role': 'user', 'content': user_parts[0]['text']})

        response = await self.ai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    @app_commands.command(name='analyze', description='Analisa um log de servidor Minecraft com IA')
    @app_commands.checks.cooldown(COOLDOWN_RATE, COOLDOWN_PER)
    @app_commands.describe(
        log_link='Link do mclo.gs ou pastebin com o log',
        log_file='Arquivo .log ou .txt para analisar',
        image='Screenshot de erro/log para análise visual (opcional)',
    )
    async def analyze(
        self,
        interaction: discord.Interaction,
        log_link: str | None = None,
        log_file: discord.Attachment | None = None,
        image: discord.Attachment | None = None,
    ):
        if not self.ai_client:
            await interaction.response.send_message(
                '⚠️ Comando indisponível: chave de API não configurada.', ephemeral=True
            )
            return

        if not log_link and not log_file and not image:
            await interaction.response.send_message(
                'Forneça um link (mclo.gs/pastebin), um arquivo (.log/.txt), ou uma imagem.', ephemeral=True
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

        # Validate link format *before* deferring so invalid links fail fast
        fetch_url = None
        mclogs_url = None
        if log_link:
            fetch_url, mclogs_url = _parse_link(log_link)
            if not fetch_url:
                await interaction.response.send_message(
                    'Link não reconhecido. Use um link do mclo.gs ou pastebin.com.',
                    ephemeral=True,
                )
                return

        await interaction.response.defer(thinking=True)

        log_content = None

        if log_file:
            if not log_file.filename.endswith(('.txt', '.log')):
                await interaction.followup.send(
                    'Formato não suportado. Envie um arquivo `.log` ou `.txt`.', ephemeral=True
                )
                return
            log_content = await self.read_file_content(log_file.url)
            if log_content:
                mclogs_url = await self.upload_mclogs(log_content)
        elif fetch_url:
            log_content = await self.read_file_content(fetch_url)

        if not log_content and not image_url:
            await interaction.followup.send('Não foi possível ler o conteúdo do log.')
            return

        try:
            analysis = await self._analyze_with_ai(log_content=log_content, image_url=image_url)
        except RateLimitError:
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
            return
        except Exception:
            logger.exception("AI log analysis failed")
            await interaction.followup.send('Ocorreu um erro ao analisar o log. Tente novamente.')
            return

        if not analysis:
            await interaction.followup.send('Ocorreu um erro ao analisar o log. Tente novamente.')
            return

        analysis = _truncate_safe(analysis)

        embed = discord.Embed(
            title='🔬 Análise de Log',
            description=analysis,
            color=discord.Color.orange(),
        )

        if log_file:
            embed.add_field(name='Arquivo', value=log_file.filename, inline=True)
        if image:
            embed.set_thumbnail(url=image.url)
        if mclogs_url:
            embed.add_field(name='mclo.gs', value=f'[Ver log]({mclogs_url})', inline=True)

        embed.set_footer(text='Análise gerada por IA • Sempre verifique manualmente')
        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(LogAnalyzer(bot))
