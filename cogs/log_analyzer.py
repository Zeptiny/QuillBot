import logging
import os
import re

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI

from responses.errors import patterns

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
CHAT_MODEL = os.getenv('CHAT_MODEL', 'google/gemini-2.5-flash-lite')

MCLO_GS_PATTERN = re.compile(r'https://mclo\.gs/\w+')
PASTEBIN_PATTERN = re.compile(r'https://pastebin\.com/(\w+)')
MAX_CONTENT_SIZE = 5 * 1024 * 1024  # 5MB limit
MAX_LOG_CONTEXT = 12000  # Max characters sent to the LLM

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


def check_message(content: str) -> str | None:
    """Check message content against known error patterns."""
    for compiled_pattern, response_template in patterns:
        match = compiled_pattern.search(content)
        if match:
            return response_template.format(*match.groups())
    return None


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

    async def read_file_content(self, url: str) -> str | None:
        """Fetch text content from a URL with size limit."""
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                content_length = resp.headers.get('Content-Length')
                if content_length and int(content_length) > MAX_CONTENT_SIZE:
                    logger.warning("Content too large from %s (%s bytes)", url, content_length)
                    return None
                return await resp.text()
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

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        response = check_message(message.content)

        # Check for paste service links
        link = None
        mclogs_match = MCLO_GS_PATTERN.search(message.content)
        pastebin_match = PASTEBIN_PATTERN.search(message.content)

        if mclogs_match:
            link = mclogs_match.group(0)
        elif pastebin_match:
            link = f'https://pastebin.com/raw/{pastebin_match.group(1)}'

        if link:
            await message.add_reaction('👀')
            link_content = await self.read_file_content(link)
            if link_content and not response:
                response = check_message(link_content)

        # Check file attachments
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.endswith(('.txt', '.log')):
                    file_content = await self.read_file_content(attachment.url)
                    if file_content:
                        if not response:
                            response = check_message(file_content)
                        upload_link = await self.upload_mclogs(file_content)
                        if upload_link:
                            await message.reply(
                                f'Na próxima vez busque utilizar um serviço para enviar suas logs, '
                                f'como o mclo.gs, fiz o upload para você <3:\n<{upload_link}>'
                            )
                        else:
                            await message.reply('Algo deu errado ao tentar fazer o upload para o mclo.gs.')
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

        try:
            response = await self.ai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("AI log analysis failed")
            return None

    @app_commands.command(name='analyze', description='Analisa um log de servidor Minecraft com IA')
    @app_commands.describe(
        log_link='Link do mclo.gs ou pastebin com o log',
        log_file='Arquivo .log ou .txt para analisar',
        imagem='Screenshot de erro/log para análise visual (opcional)',
    )
    async def analyze(
        self,
        interaction: discord.Interaction,
        log_link: str | None = None,
        log_file: discord.Attachment | None = None,
        imagem: discord.Attachment | None = None,
    ):
        if not self.ai_client:
            await interaction.response.send_message(
                '⚠️ Comando indisponível: chave de API não configurada.', ephemeral=True
            )
            return

        if not log_link and not log_file and not imagem:
            await interaction.response.send_message(
                'Forneça um link (mclo.gs/pastebin), um arquivo (.log/.txt), ou uma imagem.', ephemeral=True
            )
            return

        image_url = None
        if imagem:
            if not imagem.content_type or not imagem.content_type.startswith('image/'):
                await interaction.response.send_message(
                    'O arquivo enviado não é uma imagem válida.', ephemeral=True
                )
                return
            image_url = imagem.url

        await interaction.response.defer(thinking=True)

        log_content = None
        mclogs_url = None

        if log_file:
            if not log_file.filename.endswith(('.txt', '.log')):
                await interaction.followup.send('Formato não suportado. Envie um arquivo `.log` ou `.txt`.')
                return
            log_content = await self.read_file_content(log_file.url)
            # Also upload to mclo.gs for reference
            if log_content:
                mclogs_url = await self.upload_mclogs(log_content)
        elif log_link:
            # Parse the link
            mclogs_match = MCLO_GS_PATTERN.search(log_link)
            pastebin_match = PASTEBIN_PATTERN.search(log_link)
            if mclogs_match:
                url = mclogs_match.group(0)
                mclogs_url = url
            elif pastebin_match:
                url = f'https://pastebin.com/raw/{pastebin_match.group(1)}'
                mclogs_url = None
            else:
                await interaction.followup.send(
                    'Link não reconhecido. Use um link do mclo.gs ou pastebin.com.'
                )
                return
            log_content = await self.read_file_content(url)

        if not log_content and not image_url:
            await interaction.followup.send('Não foi possível ler o conteúdo do log.')
            return

        analysis = await self._analyze_with_ai(log_content=log_content, image_url=image_url)

        if not analysis:
            await interaction.followup.send('Ocorreu um erro ao analisar o log. Tente novamente.')
            return

        # Truncate if too long for embed
        if len(analysis) > 3800:
            analysis = analysis[:3800] + '\n\n*...análise truncada*'

        embed = discord.Embed(
            title='🔬 Análise de Log',
            description=analysis,
            color=discord.Color.orange(),
        )

        if log_file:
            embed.add_field(name='Arquivo', value=log_file.filename, inline=True)
        if imagem:
            embed.set_thumbnail(url=imagem.url)
        if mclogs_url:
            embed.add_field(name='mclo.gs', value=f'[Ver log]({mclogs_url})', inline=True)

        embed.set_footer(text='Análise gerada por IA • Sempre verifique manualmente')
        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(LogAnalyzer(bot))
