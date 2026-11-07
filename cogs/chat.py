"""Chat cog: ``/chat``, ``/ask``, and the Discord events behind conversations.

The logic lives in :mod:`core.chat_engine` (``bot.services.chat``); this cog
plugs it into Discord: the slash commands, the router handler that answers
messages, the edit/delete events, and the episode worker that turns ended
conversations into memories.
"""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import EPISODES_ENABLED, EPISODES_INTERVAL_SECONDS, MEMORY_ENABLED
from core import context_tools, radar_tools
from core.chat_engine import ChatEngine
from core.episodes import EpisodeWriter
from core.profiles import ASK, CHAT

logger = logging.getLogger(__name__)

ROUTER_NAME = 'chat'


class Chat(commands.Cog):
    """Conversations: slash commands, mentions, replies, threads and name calls."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self.engine = ChatEngine(bot, self.services)
        self.services.chat = self.engine
        context_tools.register(self.services.tools)
        radar_tools.register(self.services.tools)
        self._episode_task: asyncio.Task | None = None

    async def cog_load(self) -> None:
        self.services.router.handle(ROUTER_NAME, self.engine.handle_message, priority=10)
        if EPISODES_ENABLED and MEMORY_ENABLED:
            self._episode_task = asyncio.get_running_loop().create_task(self._episode_loop())

    async def cog_unload(self) -> None:
        self.services.tools.unregister(*radar_tools.RADAR_TOOLS)
        self.services.router.remove(ROUTER_NAME)
        await self.engine.bursts.close()
        if self._episode_task is not None:
            self._episode_task.cancel()
        if self.services.chat is self.engine:
            self.services.chat = None

    async def _episode_loop(self) -> None:
        writer = EpisodeWriter(self.services)
        await self.bot.wait_until_ready()
        while True:
            try:
                await asyncio.sleep(EPISODES_INTERVAL_SECONDS)
                if self.services.memory is not None and self.services.llm.available:
                    await writer.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('[episodes] loop error')

    # --- Slash commands ---

    @app_commands.command(name='chat', description='Converse com o bot')
    @app_commands.describe(question='Sua mensagem', image='Imagem/screenshot (opcional)')
    async def chat(self, interaction: discord.Interaction, question: str, image: discord.Attachment | None = None):
        if not await self._check(interaction, image):
            return
        await self.engine.answer_interaction(interaction, question, image=image, profile=CHAT)

    @app_commands.command(name='ask', description='Pergunte algo sobre administração de servidores Minecraft')
    @app_commands.describe(question='Sua pergunta', image='Imagem/screenshot para análise (opcional)')
    async def ask(self, interaction: discord.Interaction, question: str, image: discord.Attachment | None = None):
        docs = self.services.docs
        if docs is not None and docs.indexing:
            await interaction.response.send_message(
                '📚 A documentação está sendo indexada, tente novamente em alguns instantes.', ephemeral=True,
            )
            return
        if not await self._check(interaction, image):
            return
        await self.engine.answer_interaction(interaction, question, image=image, profile=ASK)

    async def _check(self, interaction: discord.Interaction, image: discord.Attachment | None) -> bool:
        if not self.services.llm.available:
            await interaction.response.send_message('⚠️ Comando indisponível: chave de API não configurada.', ephemeral=True)
            return False
        if image is not None and not (image.content_type or '').startswith('image/'):
            await interaction.response.send_message('O arquivo enviado não é uma imagem válida.', ephemeral=True)
            return False
        return True

    # --- Edits and deletes of messages the bot answered ---

    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
        try:
            await self.engine.message_deleted(payload.message_id, payload.channel_id, payload.guild_id)
        except Exception:
            logger.exception('[chat] delete handling failed for %s', payload.message_id)

    @commands.Cog.listener()
    async def on_raw_bulk_message_delete(self, payload: discord.RawBulkMessageDeleteEvent):
        for message_id in payload.message_ids:
            try:
                await self.engine.message_deleted(message_id, payload.channel_id, payload.guild_id)
            except Exception:
                logger.exception('[chat] delete handling failed for %s', message_id)

    @commands.Cog.listener()
    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent):
        author = (payload.data or {}).get('author') or {}
        if author.get('bot'):
            return
        try:
            await self.engine.message_edited(
                payload.message_id, payload.channel_id, payload.guild_id,
                (payload.data or {}).get('content'),
            )
        except Exception:
            logger.exception('[chat] edit handling failed for %s', payload.message_id)


async def setup(bot: commands.Bot):
    await bot.add_cog(Chat(bot))
