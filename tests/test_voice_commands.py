"""Voice command authorization, per-guild ownership and startup/stop races."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from cogs import voice
from core.realtime_voice import VoiceConversation


def setup_voice(monkeypatch):
    monkeypatch.setattr(voice, 'TOKEN_PLAN_API_KEY', 'test-key')
    channel = Mock(spec=discord.VoiceChannel)
    channel.id = 10
    channel.mention = '<#10>'
    channel.send = AsyncMock()
    channel.permissions_for.return_value = SimpleNamespace(view_channel=True, connect=True, speak=True, send_messages=True)
    guild = SimpleNamespace(id=1, me=SimpleNamespace(id=99), voice_client=None)
    user = SimpleNamespace(id=5, bot=False, guild=guild, voice=SimpleNamespace(channel=channel),
                           guild_permissions=SimpleNamespace(manage_guild=False))
    channel.members = [user]
    client = SimpleNamespace(
        channel=channel, disconnect=AsyncMock(), stop_listening=Mock(), stop_playing=Mock(),
    )
    channel.connect = AsyncMock(return_value=client)
    bot = SimpleNamespace(user=SimpleNamespace(id=99), services=SimpleNamespace(persona=SimpleNamespace(block=lambda gid: 'Persona!')))
    cog = voice.Voice(bot)
    interaction = SimpleNamespace(
        guild=guild, guild_id=1, user=user,
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    qwen = SimpleNamespace(open=AsyncMock(), close=AsyncMock())
    factory = Mock(return_value=qwen)
    monkeypatch.setattr(voice, 'QwenRealtime', factory)

    async def waiting(self):
        await asyncio.Future()

    monkeypatch.setattr(VoiceConversation, 'run', waiting)
    return cog, interaction, channel, client, qwen, factory


async def test_join_announces_before_capture_and_uses_persona(monkeypatch):
    cog, interaction, channel, client, qwen, factory = setup_voice(monkeypatch)
    await cog.join.callback(cog, interaction)
    qwen.open.assert_awaited_once()
    assert factory.call_args.kwargs['instructions'].endswith('Persona!')
    assert channel.connect.call_args.kwargs['self_deaf'] is False
    channel.send.assert_awaited_once()
    assert 'Qwen' in channel.send.call_args.args[0]
    assert 1 in cog.sessions
    await cog.leave.callback(cog, interaction)
    assert not cog.sessions and not cog._tasks
    client.disconnect.assert_awaited_once_with(force=True)
    qwen.close.assert_awaited_once()


async def test_simultaneous_join_is_serialized(monkeypatch):
    cog, interaction, channel, client, qwen, _ = setup_voice(monkeypatch)
    await asyncio.gather(cog.join.callback(cog, interaction), cog.join.callback(cog, interaction))
    channel.connect.assert_awaited_once()
    assert len(cog.sessions) == 1
    assert any('Já estou' in call.args[0] for call in interaction.followup.send.call_args_list)
    await cog.cog_unload()
    assert not cog.sessions
    client.disconnect.assert_awaited_once()


async def test_other_channel_cannot_stop_but_admin_can(monkeypatch):
    cog, interaction, channel, client, *_ = setup_voice(monkeypatch)
    await cog.join.callback(cog, interaction)
    interaction.user.voice.channel = SimpleNamespace(id=20)
    await cog.leave.callback(cog, interaction)
    assert 1 in cog.sessions
    assert 'mesmo canal' in interaction.followup.send.call_args.args[0]
    client.disconnect.assert_not_awaited()
    interaction.user.guild_permissions.manage_guild = True
    await cog.leave.callback(cog, interaction)
    assert not cog.sessions
    client.disconnect.assert_awaited_once()


@pytest.mark.parametrize('failure', ['qwen', 'discord', 'announcement', 'user_left'])
async def test_failed_start_does_not_leave_a_session_or_socket(monkeypatch, failure, caplog):
    cog, interaction, channel, client, qwen, _ = setup_voice(monkeypatch)
    error = RuntimeError('test-key secret-provider-payload')
    if failure == 'qwen':
        qwen.open.side_effect = error
    elif failure == 'discord':
        async def connect(**kwargs):
            interaction.guild.voice_client = client
            raise error
        channel.connect.side_effect = connect
    elif failure == 'announcement':
        channel.send.side_effect = error
    else:
        async def open_qwen():
            interaction.user.voice.channel = None
        qwen.open.side_effect = open_qwen
    await cog.join.callback(cog, interaction)
    assert not cog.sessions and not cog._tasks
    qwen.close.assert_awaited_once()
    if failure != 'qwen':
        client.disconnect.assert_awaited_once()
    output = str(interaction.followup.send.call_args) + caplog.text
    assert 'test-key' not in output
    assert 'secret-provider-payload' not in output


@pytest.mark.parametrize('missing', ['key', 'channel', 'permissions', 'guild'])
async def test_invalid_join_never_connects(monkeypatch, missing):
    cog, interaction, channel, _, qwen, _ = setup_voice(monkeypatch)
    if missing == 'key':
        monkeypatch.setattr(voice, 'TOKEN_PLAN_API_KEY', None)
    elif missing == 'channel':
        interaction.user.voice.channel = None
    elif missing == 'permissions':
        channel.permissions_for.return_value.speak = False
    else:
        interaction.guild = None
    await cog.join.callback(cog, interaction)
    qwen.open.assert_not_awaited()
    channel.connect.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once()


@pytest.mark.parametrize('event', ['empty', 'moved', 'disconnected'])
async def test_empty_channel_or_bot_move_ends_conversation(monkeypatch, event):
    cog, interaction, channel, client, *_ = setup_voice(monkeypatch)
    await cog.join.callback(cog, interaction)
    channel.members = []
    member = interaction.user
    if event != 'empty':
        member = SimpleNamespace(id=99, guild=interaction.guild)
    after = SimpleNamespace(channel=SimpleNamespace(id=50) if event == 'moved' else None)
    await cog.on_voice_state_update(member, SimpleNamespace(channel=channel), after)
    assert not cog.sessions
    client.disconnect.assert_awaited_once()


async def test_unload_waits_for_startup_then_closes(monkeypatch):
    cog, interaction, channel, client, qwen, _ = setup_voice(monkeypatch)
    gate = asyncio.Event()
    qwen.open.side_effect = gate.wait
    joining = asyncio.create_task(cog.join.callback(cog, interaction))
    await asyncio.sleep(0)
    unloading = asyncio.create_task(cog.cog_unload())
    await asyncio.sleep(0)
    assert not unloading.done()
    gate.set()
    await joining
    await unloading
    assert not cog.sessions and not cog._tasks
    client.disconnect.assert_awaited_once()
