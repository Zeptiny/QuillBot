"""``/voz`` — real-time voice conversation in voice channels.

``/voz entrar`` brings the bot into the caller's voice channel and connects it
to a speech-to-speech model (Qwen-Audio realtime, :mod:`core.realtime_voice`);
``/voz sair`` ends the call. Everything that happens during the call — audio
in and out, barge-in, transcripts, ending idle calls — is
:class:`core.voice_call.VoiceCall`.

Loaded only when the bot is configured for it (``VOICE_ENABLED``,
``VOICE_API_KEY``, ``VOICE_REALTIME_URL``, libopus and davey present), and a
server still has to opt in (``/config servidor`` → Conversa por voz), since a
call's audio goes to that model.
"""

from __future__ import annotations

import datetime
import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import (
    BR_TZ,
    VOICE_API_KEY,
    VOICE_AVAILABLE,
    VOICE_BARGE_IN_MS,
    VOICE_CAN_STAY_SILENT,
    VOICE_ENABLED,
    VOICE_IDLE_MINUTES,
    VOICE_MAX_CALLS,
    VOICE_MAX_MINUTES,
    VOICE_MODEL,
    VOICE_REALTIME_URL,
    VOICE_TRANSCRIPTS,
    VOICE_TURN_DETECTION,
    VOICE_VOICE,
    VOICE_WAKE_WORDS,
    VOICE_WEB_SEARCH,
)
from core.prompts import VOICE_SILENCE_TOOL, voice_greeting, voice_instructions
from core.realtime_voice import RealtimeError, RealtimeVoiceClient
from core.voice_call import VoiceCall

logger = logging.getLogger(__name__)


def can_stay_silent() -> bool:
    """Whether the model gets the silence tool (Qwen-Audio 3.0 takes tools or web search, not both)."""
    return VOICE_CAN_STAY_SILENT and not VOICE_WEB_SEARCH


def session_config(instructions: str) -> dict:
    """The realtime session the bot asks for (``session.update``)."""
    config: dict = {
        'modalities': ['audio', 'text'],
        'voice': VOICE_VOICE,
        'instructions': instructions,
        'input_audio_format': 'pcm',
        'output_audio_format': 'pcm',
        'turn_detection': {'type': VOICE_TURN_DETECTION},
    }
    if VOICE_WEB_SEARCH:
        config['enable_search'] = True
    elif can_stay_silent():
        config['tools'] = [VOICE_SILENCE_TOOL]
    return config


def missing_dependency() -> str | None:
    """What keeps this machine from doing voice, if anything."""
    try:
        import davey  # noqa: F401
        import nacl.bindings  # noqa: F401
    except ImportError as e:
        return f'{e.name} is not installed (pip install -r requirements.txt)'
    try:
        discord.opus.Decoder()  # loads libopus
    except discord.opus.OpusNotLoaded:
        return 'libopus was not found (apt install libopus0)'
    return None


class _JoinError(Exception):
    """Why the bot couldn't join (user-facing)."""


class Voice(commands.Cog):
    """Voice calls: /voz entrar | sair."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self.calls: dict[int, VoiceCall] = {}
        self._joining: set[int] = set()

    voz = app_commands.Group(name='voz', description='Conversa por voz com o bot nas calls', guild_only=True)

    async def cog_unload(self) -> None:
        for call in list(self.calls.values()):
            await call.end('o bot está reiniciando')

    # --- commands ---

    @voz.command(name='entrar', description='Chama o bot para a call em que você está')
    async def entrar(self, interaction: discord.Interaction):
        guild = interaction.guild
        member = interaction.user
        assert guild is not None and isinstance(member, discord.Member)
        problem = self._join_problem(guild, member)
        if problem:
            await interaction.response.send_message(problem, ephemeral=True)
            return
        channel = member.voice.channel if member.voice else None
        assert isinstance(channel, discord.VoiceChannel)  # _join_problem made sure
        await interaction.response.defer(thinking=True)
        self._joining.add(guild.id)
        try:
            call = await self._start_call(guild, channel, member)
        except _JoinError as e:
            await interaction.followup.send(f'⚠️ {e}')
            return
        finally:
            self._joining.discard(guild.id)
        transcribed = ' e eu escrevo a conversa no chat da call' if call.transcripts else ''
        wake = f' Eu só respondo quando alguém me chama de “{VOICE_WAKE_WORDS[0]}”.' if VOICE_WAKE_WORDS else ''
        await interaction.followup.send(
            f'🎙️ Entrei em {channel.mention}. Enquanto eu estiver lá, o que vocês falarem vai para a '
            f'IA de voz (Qwen, da Alibaba Cloud){transcribed}.{wake} Para me tirar: `/voz sair`.'
        )

    @voz.command(name='sair', description='Tira o bot da call')
    async def sair(self, interaction: discord.Interaction):
        guild = interaction.guild
        member = interaction.user
        assert guild is not None and isinstance(member, discord.Member)
        call = self.calls.get(guild.id)
        if call is None:
            stray = guild.voice_client
            if stray is not None:  # a connection that outlived its call
                await stray.disconnect(force=True)
                await interaction.response.send_message('👋 Saí da call.', ephemeral=True)
                return
            await interaction.response.send_message('Não estou em nenhuma call aqui.', ephemeral=True)
            return
        in_call = member.voice is not None and member.voice.channel == call.channel
        perms = member.guild_permissions
        if not (in_call or perms.move_members or perms.manage_guild):
            await interaction.response.send_message(
                'Só quem está na call (ou quem pode mover membros) me tira de lá.', ephemeral=True,
            )
            return
        await interaction.response.send_message(f'👋 Saindo de {call.channel.mention}.')
        await call.end(f'pedido de {member.display_name}')

    # --- joining ---

    def _join_problem(self, guild: discord.Guild, member: discord.Member) -> str | None:
        if not self.services.settings.get(guild.id, None, 'voice'):
            perms = member.guild_permissions
            how = (
                'Ligue com `/config servidor opcao:Conversa por voz valor:sim`.'
                if perms.manage_guild or perms.administrator
                else 'Um admin pode ligar em `/config servidor`.'
            )
            return f'A conversa por voz está desligada neste servidor. {how}'
        channel = member.voice.channel if member.voice else None
        if channel is None:
            return 'Entre numa call e use `/voz entrar` de novo.'
        if isinstance(channel, discord.StageChannel):
            return 'Em palcos (stage) eu ainda não falo, só em canais de voz comuns.'
        call = self.calls.get(guild.id)
        if call is not None:
            if call.ended:
                return 'Ainda estou saindo da call anterior; tente de novo em alguns segundos.'
            if call.channel == channel:
                return 'Já estou nesta call.'
            return f'Já estou em outra call deste servidor: {call.channel.mention}.'
        if guild.id in self._joining:
            return 'Já estou entrando numa call…'
        if len(self.calls) + len(self._joining) >= VOICE_MAX_CALLS:
            return 'Já estou no máximo de calls ao mesmo tempo. Tente de novo mais tarde.'
        perms = channel.permissions_for(guild.me)
        missing = [name for name, ok in (('Conectar', perms.connect), ('Falar', perms.speak)) if not ok]
        if missing:
            return f'Não tenho permissão de {" e ".join(missing)} em {channel.mention}.'
        if channel.user_limit and len(channel.members) >= channel.user_limit and not perms.move_members:
            return f'{channel.mention} está cheia.'
        return None

    async def _start_call(
        self, guild: discord.Guild, channel: discord.VoiceChannel, member: discord.Member,
    ) -> VoiceCall:
        from core.voice_receive import ListeningVoiceClient

        participants = [m.display_name for m in channel.members if not m.bot]
        persona = self.services.persona.block(guild.id) if self.services.persona is not None else ''
        instructions = voice_instructions(
            server=guild.name,
            channel=channel.name,
            caller=member.display_name,
            participants=participants,
            now=datetime.datetime.now(BR_TZ).strftime('%d/%m/%Y %H:%M'),
            name=VOICE_WAKE_WORDS[0] if VOICE_WAKE_WORDS else guild.me.display_name,
            persona=persona,
            can_stay_silent=can_stay_silent(),
        )
        assert VOICE_API_KEY is not None
        realtime = RealtimeVoiceClient(VOICE_REALTIME_URL, VOICE_API_KEY, VOICE_MODEL)
        try:
            await realtime.connect(session_config(instructions))
        except RealtimeError as e:
            logger.warning('[voice] realtime session failed in guild %s: %s', guild.id, e)
            raise _JoinError(f'Não consegui abrir a conversa com a IA de voz: {e}.') from None

        if guild.voice_client is not None:  # a connection that outlived its call
            await guild.voice_client.disconnect(force=True)
        try:
            vc = await channel.connect(cls=ListeningVoiceClient, timeout=20, self_deaf=False)
        except Exception as e:
            await realtime.close()
            logger.warning('[voice] could not join #%s (%s): %r', channel.name, guild.id, e)
            raise _JoinError(f'Não consegui entrar em {channel.mention} ({type(e).__name__}).') from None

        call = VoiceCall(
            voice_client=vc,
            realtime=realtime,
            is_bot=lambda uid: bool(getattr(guild.get_member(uid), 'bot', False)),
            name_of=lambda uid: getattr(guild.get_member(uid), 'display_name', None) or str(uid),
            transcripts=VOICE_TRANSCRIPTS,
            barge_in_ms=VOICE_BARGE_IN_MS,
            wake_words=VOICE_WAKE_WORDS,
            idle_seconds=VOICE_IDLE_MINUTES * 60,
            max_seconds=VOICE_MAX_MINUTES * 60,
            on_end=lambda ended: self._forget(guild.id, ended),
        )
        self.calls[guild.id] = call
        try:
            await call.start(greeting=voice_greeting(
                member.display_name, VOICE_WAKE_WORDS[0] if VOICE_WAKE_WORDS else None,
            ))
        except Exception:
            logger.exception('[voice] starting the call in #%s (%s) failed', channel.name, guild.id)
            await call.end('não consegui começar a conversa')
            raise _JoinError('Entrei, mas não consegui começar a conversa.') from None
        logger.info('[voice] call started in #%s (%s) by %s', channel.name, guild.id, member)
        return call

    def _forget(self, guild_id: int, call: VoiceCall) -> None:
        if self.calls.get(guild_id) is call:
            del self.calls[guild_id]

    # --- the bot's own voice state ---

    @commands.Cog.listener()
    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState,
    ):
        call = self.calls.get(member.guild.id)
        if call is None or call.ended:
            return
        me = self.bot.user
        if me is not None and member.id == me.id:
            if after.channel is None:
                await call.end('fui desconectado da call')
            return
        if before.channel == call.channel and after.channel != call.channel:
            if not any(not m.bot for m in call.channel.members):
                await call.end('todo mundo saiu da call')


async def setup(bot: commands.Bot):
    if not VOICE_ENABLED:
        logger.info('Voice calls disabled via VOICE_ENABLED=false')
        return
    if not VOICE_AVAILABLE:
        logger.warning('Voice calls disabled: VOICE_API_KEY and VOICE_REALTIME_URL must both be set')
        return
    missing = missing_dependency()
    if missing:
        logger.error('Voice calls disabled: %s', missing)
        return
    await bot.add_cog(Voice(bot))
