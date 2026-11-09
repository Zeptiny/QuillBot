"""/voice entrar, sair and status — real-time conversations in Discord voice."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import suppress

import discord
from discord import app_commands
from discord.ext import commands, voice_recv

from config import (
    QWEN_REALTIME_MODEL,
    QWEN_REALTIME_URL,
    QWEN_REALTIME_VOICE,
    TOKEN_PLAN_API_KEY,
    VOICE_ENABLED,
    VOICE_IDLE_TIMEOUT_SECONDS,
    VOICE_MAX_SESSION_SECONDS,
)
from core.realtime_voice import QwenRealtime, VoiceConversation, VoiceError

logger = logging.getLogger(__name__)


def voice_instructions(persona: str) -> str:
    return (
        'Você é QuillBot, conversando ao vivo com as pessoas em um canal de voz do Discord. '
        'Responda de forma breve e natural, em português brasileiro por padrão, acompanhando '
        'o idioma de quem fala. Pode haver várias pessoas; não invente quem disse algo. '
        'Esta sessão tem apenas áudio: você não tem acesso a ferramentas, mensagens de texto, '
        'pesquisa ou memórias de outras conversas. Não diga que executou ações no servidor.\n\n'
        + persona
    )


class Voice(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.sessions: dict[int, VoiceConversation] = {}
        self._tasks: dict[int, asyncio.Task] = {}
        self._locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._unloading = False

    voice = app_commands.Group(name='voice', description='Converse com o bot por voz', guild_only=True)

    @voice.command(name='entrar', description='Entra no seu canal e conversa por voz com as pessoas presentes')
    async def join(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        channel = getattr(getattr(interaction.user, 'voice', None), 'channel', None)
        if guild is None or not isinstance(channel, discord.VoiceChannel):
            await interaction.response.send_message('Entre em um canal de voz comum primeiro (não um palco).', ephemeral=True)
            return
        if not TOKEN_PLAN_API_KEY:
            await interaction.response.send_message(
                'Configure TOKEN_PLAN_API_KEY (ou ANTHROPIC_AUTH_TOKEN) no ambiente do bot e reinicie-o.',
                ephemeral=True,
            )
            return
        permissions = channel.permissions_for(guild.me)
        if not (permissions.view_channel and permissions.connect and permissions.speak and permissions.send_messages):
            await interaction.response.send_message(
                'Preciso de Ver canal, Conectar, Falar e Enviar mensagens no canal de voz.', ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        async with self._locks[guild.id]:
            if self._unloading:
                await interaction.followup.send('O serviço de voz está encerrando.', ephemeral=True)
                return
            if guild.id in self.sessions or guild.voice_client is not None:
                await interaction.followup.send('Já estou em um canal de voz neste servidor. Use /voice sair primeiro.', ephemeral=True)
                return
            qwen = None
            conversation = None
            client = None
            try:
                # Fail before joining if the system Opus library is missing.
                try:
                    discord.opus.Decoder()
                except discord.opus.OpusNotLoaded:
                    raise VoiceError('Instale a biblioteca libopus no servidor do bot e reinicie-o.') from None
                persona = self.bot.services.persona
                qwen = QwenRealtime(
                    api_key=TOKEN_PLAN_API_KEY, url=QWEN_REALTIME_URL,
                    model=QWEN_REALTIME_MODEL, voice=QWEN_REALTIME_VOICE,
                    instructions=voice_instructions(persona.block(guild.id) if persona else ''),
                )
                await qwen.open()
                client = await channel.connect(
                    cls=voice_recv.VoiceRecvClient, timeout=20, reconnect=False, self_deaf=False,
                )
                conversation = VoiceConversation(
                    client, qwen, idle_timeout=VOICE_IDLE_TIMEOUT_SECONDS, max_duration=VOICE_MAX_SESSION_SECONDS,
                )
                if getattr(getattr(interaction.user, 'voice', None), 'channel', None) != channel:
                    raise VoiceError('Você saiu do canal durante a conexão. Use /voice entrar novamente.')
                # Announce in the voice channel itself, before capturing audio.
                await channel.send(
                    '🎙️ Conversa por voz iniciada. O áudio das pessoas neste canal será enviado ao Qwen '
                    'para gerar respostas faladas. Não salvo áudio nem transcrições no bot. '
                    'Falem um de cada vez; usem fones para evitar eco. '
                    'Quem estiver aqui pode encerrar com `/voice sair`.',
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except BaseException as error:
                try:
                    if conversation is not None:
                        await conversation.close()
                    else:
                        try:
                            if qwen is not None:
                                await qwen.close()
                        finally:
                            # connect() can fail after registering a voice client.
                            client = client or guild.voice_client
                            if client is not None:
                                await client.disconnect(force=True)
                except Exception as cleanup_error:
                    self._failure_message(cleanup_error)
                if isinstance(error, asyncio.CancelledError):
                    raise
                if not isinstance(error, Exception):
                    raise
                await interaction.followup.send(self._failure_message(error), ephemeral=True)
                return
            self.sessions[guild.id] = conversation
            self._tasks[guild.id] = asyncio.create_task(self._run(guild.id, conversation))
        await interaction.followup.send(f'Conectado a {channel.mention}. Pode começar a falar! Use /voice sair para encerrar.', ephemeral=True)

    @staticmethod
    def _failure_message(error: Exception) -> str:
        # No exception traceback or provider message: those may contain secrets,
        # response payloads, audio or transcripts even with DEBUG logging on.
        logger.warning('Voice session failed (%s)', type(error).__name__)
        if isinstance(error, VoiceError):
            return str(error)
        return (
            'Não consegui manter a conexão de voz. Confira a chave Token Plan, a região, '
            'as dependências de voz e as permissões do bot; depois use /voice entrar novamente.'
        )

    async def _run(self, guild_id: int, conversation: VoiceConversation) -> None:
        reason = None
        try:
            reason = await conversation.run()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            reason = self._failure_message(error)
        finally:
            try:
                await conversation.close()
            except Exception as error:
                reason = self._failure_message(error)
            finally:
                if self.sessions.get(guild_id) is conversation:
                    self.sessions.pop(guild_id, None)
                    self._tasks.pop(guild_id, None)
        if reason and not self._unloading:
            with suppress(discord.HTTPException):
                await conversation.channel.send(reason, allowed_mentions=discord.AllowedMentions.none())

    async def _stop(self, guild_id: int) -> None:
        conversation = self.sessions.get(guild_id)
        task = self._tasks.get(guild_id)
        try:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        finally:
            if conversation is not None:
                # A task cancelled before its first tick never enters its finally.
                try:
                    await conversation.close()
                except Exception as error:
                    self._failure_message(error)
                finally:
                    if self.sessions.get(guild_id) is conversation:
                        self.sessions.pop(guild_id, None)
                        self._tasks.pop(guild_id, None)

    @voice.command(name='sair', description='Encerra a conversa e sai do canal de voz')
    async def leave(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message('Use este comando em um servidor.', ephemeral=True)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        async with self._locks[interaction.guild.id]:
            conversation = self.sessions.get(interaction.guild.id)
            channel = getattr(getattr(interaction.user, 'voice', None), 'channel', None)
            if conversation is None:
                await interaction.followup.send('Não estou em uma conversa por voz neste servidor.', ephemeral=True)
                return
            if channel != conversation.channel and not interaction.user.guild_permissions.manage_guild:
                await interaction.followup.send(
                    'Entre no mesmo canal de voz ou peça a alguém com Gerenciar Servidor para encerrar.', ephemeral=True,
                )
                return
            await self._stop(interaction.guild.id)
        await interaction.followup.send('Conversa por voz encerrada.', ephemeral=True)

    @voice.command(name='status', description='Mostra se há uma conversa por voz neste servidor')
    async def status(self, interaction: discord.Interaction) -> None:
        conversation = self.sessions.get(interaction.guild_id)
        message = (
            f'🎙️ Conversando em {conversation.channel.mention}. Use /voice sair para encerrar.'
            if conversation else 'Não há conversa por voz ativa neste servidor.'
        )
        await interaction.response.send_message(message, ephemeral=True)

    @commands.Cog.listener()
    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState,
    ) -> None:
        guild_id = member.guild.id
        conversation = self.sessions.get(guild_id)
        if conversation is None or conversation.closing:
            return
        channel = conversation.channel
        if before.channel == channel and after.channel != channel:
            conversation.input.remove_user(member.id)
        bot_moved = member.id == self.bot.user.id and after.channel != channel
        empty = before.channel == channel and not any(not user.bot for user in channel.members)
        if bot_moved or empty:
            async with self._locks[guild_id]:
                if self.sessions.get(guild_id) is conversation:
                    await self._stop(guild_id)

    async def cog_unload(self) -> None:
        self._unloading = True
        # Include locks for joins still waiting on a handshake.
        for guild_id in list(self._locks):
            async with self._locks[guild_id]:
                await self._stop(guild_id)


async def setup(bot: commands.Bot) -> None:
    if VOICE_ENABLED:
        await bot.add_cog(Voice(bot))
