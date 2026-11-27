"""Action GIFs: ``/hug``, ``/kiss``, ``/slap``… and the bot doing them too.

Each command posts an anime GIF of the caller doing the action to someone
(:mod:`core.action_gifs` has the actions, where GIFs come from, the counts and
the wording), with a Retribuir/Revidar button that only that person can press,
once. Doing an action to the bot gets an answer in character from the chat
engine (``ACTIONS_BOT_REPLY``), and the ``send_action_gif`` tool lets the bot
do them itself (``ACTIONS_TOOL_ENABLED``). A channel turns all of it off with
the ``actions`` setting (``/config``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import discord
from cachetools import TTLCache
from discord import app_commands
from discord.ext import commands

from config import (
    ACTIONS_BOT_REPLY,
    ACTIONS_COOLDOWN_SECONDS,
    ACTIONS_DB_PATH,
    ACTIONS_ENABLED,
    ACTIONS_PING_TARGET,
    ACTIONS_TOOL_ENABLED,
    ACTIONS_USER_AGENT,
)
from core import action_gifs
from core.action_gifs import ACTIONS, AFFECTION, Action, ActionStore, GifPicker
from core.tools import Tool, tool_schema
from core.turn import NO_REPLY_TOKEN

logger = logging.getLogger(__name__)

TOOL_NAME = 'send_action_gif'
# How long the Retribuir button stays up (a slash command's messages can't be edited after 15 min).
RETURN_SECONDS = 600
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=4)

Sender = Callable[..., Awaitable[Any]]

SEND_ACTION_TOOL = tool_schema(
    TOOL_NAME,
    'Manda no canal atual um GIF de anime de você fazendo uma ação com alguém, como os comandos /hug, '
    '/slap etc.: ' + ', '.join(f'{a.key} ({a.past})' for a in ACTIONS.values()) + '. Use quando pedirem '
    '("dá um abraço no Bruno", "bate nele") ou quando combinar com o momento e com a sua personalidade; '
    'no máximo um por resposta. O GIF sai antes da sua resposta, que pode ser curta ou nenhuma.',
    {
        'action': {'type': 'string', 'enum': list(ACTIONS), 'description': 'A ação.'},
        'user': {
            'type': 'string',
            'description': 'Quem recebe: o author_id da pessoa (de preferência), <@id> ou o nome.',
        },
    },
    ['action', 'user'],
)


def tool_status(args: dict, ctx: Any) -> str:
    action = ACTIONS.get(str(args.get('action') or ''))
    return f'{action.emoji} Mandando um GIF de {action.key}' if action else '🎞️ Mandando um GIF'


def followup(interaction: discord.Interaction) -> Sender:
    """Sends a message as *interaction*'s answer and returns it."""
    async def send(**kwargs: Any) -> Any:
        return await interaction.followup.send(wait=True, **kwargs)
    return send


class ReturnView(discord.ui.View):
    """The Retribuir/Revidar button under an action: only its target may press it, once."""

    def __init__(self, cog: Actions, action: Action, *, actor: Any, target: Any):
        super().__init__(timeout=RETURN_SECONDS)
        self.cog = cog
        self.action = action
        self.actor = actor
        self.target = target
        self.message: Any = None
        self.done = False
        button: discord.ui.Button = discord.ui.Button(
            label=action.return_label, emoji=action.emoji, style=discord.ButtonStyle.secondary,
        )
        button.callback = self.pressed
        self.add_item(button)

    async def pressed(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.target.id:
            await interaction.response.send_message(
                f'Só {self.target.display_name} pode {self.action.return_label.lower()}.', ephemeral=True,
            )
            return
        if self.done:
            await interaction.response.send_message('Já foi.', ephemeral=True)
            return
        if not self.cog.enabled_in(interaction.guild, interaction.channel):
            await interaction.response.send_message(
                'Os comandos de ação foram desligados neste canal.', ephemeral=True,
            )
            return
        self.done = True
        self.stop()
        await interaction.response.defer()
        try:
            await interaction.edit_original_response(view=None)
        except discord.HTTPException:
            logger.debug('could not remove the return button', exc_info=True)
        try:
            await self.cog.perform(
                self.action, actor=interaction.user, target=self.actor, guild=interaction.guild,
                channel=interaction.channel, send=followup(interaction), returned=True,
            )
        except discord.HTTPException:
            logger.exception('[actions] could not send the return of %s', self.action.key)

    async def on_timeout(self) -> None:
        if self.message is None or self.done:
            return
        try:
            await self.message.edit(view=None)
        except discord.HTTPException:
            logger.debug('could not remove an expired return button', exc_info=True)


class Actions(commands.Cog):
    """/hug, /kiss, /slap… (one command per action in core.action_gifs.ACTIONS)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self.store = ActionStore(ACTIONS_DB_PATH)
        self.session: aiohttp.ClientSession | None = None
        self.gifs: GifPicker | None = None
        self._last_use: TTLCache = TTLCache(maxsize=4096, ttl=max(ACTIONS_COOLDOWN_SECONDS, 1.0))
        self._commands: list[app_commands.Command] = []

    async def cog_load(self) -> None:
        await asyncio.to_thread(self.store.load)
        for action in ACTIONS.values():
            command = self.make_command(action)
            self.bot.tree.add_command(command)
            self._commands.append(command)
        if ACTIONS_TOOL_ENABLED:
            self.services.tools.register(Tool(TOOL_NAME, SEND_ACTION_TOOL, self.run_tool, tool_status))

    async def cog_unload(self) -> None:
        for command in self._commands:
            self.bot.tree.remove_command(command.name)
        self._commands.clear()
        self.services.tools.unregister(TOOL_NAME)
        if self.session is not None:
            await self.session.close()

    def user_agent(self) -> str:
        """How the GIF APIs see the bot: ``ACTIONS_USER_AGENT``, or its name and invite link."""
        if ACTIONS_USER_AGENT:
            return ACTIONS_USER_AGENT
        user = getattr(self.bot, 'user', None)
        name = ''.join((getattr(user, 'name', None) or 'QuillBot').split())
        app_id = getattr(self.bot, 'application_id', None) or getattr(user, 'id', None)
        return f'{name} (https://discord.com/oauth2/authorize?client_id={app_id})' if app_id else name

    def picker(self) -> GifPicker:
        """The GIF picker, made on first use: the bot only knows its own id once logged in."""
        if self.gifs is None:
            self.session = aiohttp.ClientSession(timeout=_HTTP_TIMEOUT, headers={'User-Agent': self.user_agent()})
            self.gifs = GifPicker(self.session, self.store)
        return self.gifs

    def make_command(self, action: Action) -> app_commands.Command:
        """The slash command for *action* (``/hug usuario:@Bruno``)."""
        async def callback(interaction: discord.Interaction, usuario: discord.Member) -> None:
            await self.use(interaction, action, usuario)

        callback = app_commands.describe(usuario='Quem recebe')(callback)
        callback = app_commands.guild_only()(callback)
        return app_commands.Command(
            name=action.key, description=f'{action.emoji} {action.description}',
            callback=callback, extras={'action': action.key},
        )

    # ------------------------------------------------------------------

    def is_me(self, person: Any) -> bool:
        me = getattr(self.bot, 'user', None)
        return me is not None and getattr(person, 'id', None) == me.id

    def enabled_in(self, guild: Any, channel: Any) -> bool:
        return guild is not None and bool(self.services.settings.get(guild.id, channel, 'actions'))

    def _wait(self, guild_id: int, user_id: int) -> float:
        last = self._last_use.get((guild_id, user_id))
        return 0.0 if last is None else max(0.0, ACTIONS_COOLDOWN_SECONDS - (time.monotonic() - last))

    @staticmethod
    def _mentions(target: Any) -> discord.AllowedMentions:
        if ACTIONS_PING_TARGET and not getattr(target, 'bot', False):
            return discord.AllowedMentions(everyone=False, roles=False, users=[target], replied_user=False)
        return discord.AllowedMentions.none()

    async def use(self, interaction: discord.Interaction, action: Action, target: Any) -> None:
        """Someone ran ``/<action> usuario:<target>``."""
        guild, channel, user = interaction.guild, interaction.channel, interaction.user
        if not self.enabled_in(guild, channel):
            await interaction.response.send_message('Os comandos de ação estão desligados neste canal.', ephemeral=True)
            return
        wait = self._wait(guild.id, user.id)
        if wait > 0:
            await interaction.response.send_message(f'⏳ Aguarde {wait:.0f}s antes de outra ação.', ephemeral=True)
            return
        actor, note = user, ''
        if target.id == user.id:
            if action.kind != AFFECTION:
                await interaction.response.send_message('Em você mesmo? Escolha outra pessoa 😅', ephemeral=True)
                return
            actor, note = guild.me or self.bot.user, 'Ninguém merece fazer isso sozinho'
        self._last_use[(guild.id, user.id)] = time.monotonic()
        await interaction.response.defer()
        try:
            await self.perform(
                action, actor=actor, target=target, guild=guild, channel=channel,
                send=followup(interaction), note=note,
            )
        except discord.HTTPException:
            logger.exception('[actions] could not send /%s', action.key)
            try:
                await interaction.followup.send('Não consegui mandar a ação aqui.', ephemeral=True)
            except discord.HTTPException:
                pass

    async def perform(
        self, action: Action, *, actor: Any, target: Any, guild: Any, channel: Any, send: Sender,
        note: str = '', returned: bool = False,
    ) -> Any:
        """Post *actor* doing *action* to *target* with *send*; returns the message.

        Under it goes a button for *target* to do it back, unless this already
        is the return or *target* is a bot. Done to this bot, the bot then
        answers in character.
        """
        gif = await self.picker().pick(action)
        count = await asyncio.to_thread(self.store.bump, guild.id, actor.id, target.id, action.key)
        content, embed = action_gifs.render(action, actor, target, gif, count, note=note, returned=returned)
        kwargs: dict[str, Any] = {'content': content, 'allowed_mentions': self._mentions(target)}
        if embed is not None:
            kwargs['embed'] = embed
        view = None
        if not returned and not getattr(target, 'bot', False):
            view = kwargs['view'] = ReturnView(self, action, actor=actor, target=target)
        sent = await send(**kwargs)
        if view is not None:
            view.message = sent
        if self.is_me(target):
            await self.answer_in_character(sent, action, actor=actor, guild=guild, channel=channel, returned=returned)
        return sent

    async def answer_in_character(
        self, sent: Any, action: Action, *, actor: Any, guild: Any, channel: Any, returned: bool,
    ) -> None:
        """The bot's own reaction to *actor* doing *action* to it (``ACTIONS_BOT_REPLY``)."""
        chat = self.services.chat
        if not ACTIONS_BOT_REPLY or chat is None or not self.services.llm.available:
            return
        if self.services.settings.get(guild.id, channel, 'respond') == 'off':
            return  # the bot doesn't converse in this channel
        # A message from the channel (not the interaction's webhook): it knows its guild and jump URL.
        anchor = channel.get_partial_message(sent.id) if hasattr(channel, 'get_partial_message') else sent
        question = action_gifs.action_question(
            action, actor, returned=returned, can_send=TOOL_NAME in self.services.tools,
        )
        try:
            await chat.answer_action(anchor, actor=actor, guild=guild, channel=channel, question=question)
        except Exception:
            logger.exception('[actions] answering /%s in character failed', action.key)

    async def run_tool(self, ctx: Any, args: dict) -> str:
        """``send_action_gif``: the bot does an action to someone, in the turn's channel."""
        guild, channel = ctx.guild, ctx.channel
        if guild is None or channel is None or not hasattr(channel, 'send'):
            return 'GIFs de ação só funcionam num canal de servidor.'
        if not self.enabled_in(guild, channel):
            return 'Os GIFs de ação estão desligados neste canal (/config); não mande.'
        if ctx.state.get(TOOL_NAME):
            return 'Você já mandou um GIF de ação nesta resposta; não mande outro.'
        action = ACTIONS.get(str(args.get('action') or '').strip().lower())
        if action is None:
            return f'Ação desconhecida: {args.get("action")}. Use uma de: {", ".join(ACTIONS)}.'
        target = action_gifs.find_member(guild, args.get('user'))
        if target is None:
            return f'Não encontrei "{args.get("user")}" neste servidor: use o author_id da pessoa (ou find_user).'
        if self.is_me(target):
            return 'Você não pode fazer uma ação em si mesmo.'
        me = guild.me or self.bot.user
        try:
            sent = await self.perform(action, actor=me, target=target, guild=guild, channel=channel, send=channel.send)
        except discord.HTTPException as e:
            return f'Não foi possível mandar o GIF: {e.text or e}'
        ctx.state[TOOL_NAME] = True
        return (
            f'GIF enviado no canal: {me.display_name} {action.past} {target.display_name} (msg_id={sent.id}). '
            f'Ele já aparece no chat: não o descreva nem repita a ação; responda curto ou {NO_REPLY_TOKEN}.'
        )


async def setup(bot: commands.Bot):
    if not ACTIONS_ENABLED:
        logger.info('Action GIFs disabled via ACTIONS_ENABLED=false')
        return
    await bot.add_cog(Actions(bot))
