"""The chat engine: when the bot talks, about what, and where the answer goes.

Every way of talking to the bot ends here: replies to its messages,
@mentions, being called by name, messages in its threads, monitors, scheduled
tasks, ``/chat``, ``/ask`` and Spark analyses. It decides whether a message is
for the bot, merges quick consecutive messages (:mod:`core.burst`), checks the
budget, finds or starts the conversation, runs the turn
(:func:`core.runner.run_turn`), sends the answer (:mod:`core.delivery`) and
stores the turn. Deleting a message the bot answered deletes the answer, and
editing it shortly after re-answers it.

Other features reach it as ``bot.services.chat``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import discord
from cachetools import TTLCache
from openai import RateLimitError

from config import (
    BR_TZ,
    CHAT_BURST_SECONDS,
    CHAT_BURST_SHORT_SECONDS,
    CHAT_DELETE_WITH_TRIGGER,
    CHAT_EDIT_WINDOW_SECONDS,
    CHAT_GUILD_BUDGET,
    CHAT_GUILD_BUDGET_WINDOW,
    CHAT_MENTION_CONTINUE_ENABLED,
    CHAT_MENTION_CONTINUE_WINDOW,
    CHAT_MENTION_ENABLED,
    CHAT_NAME_ALIASES,
    CHAT_THREAD_TITLE_MODEL,
    CHAT_USER_BUDGET,
    CHAT_USER_BUDGET_WINDOW,
    CONVERSATIONS_MAX_TURNS,
    CONVERSATIONS_TTL_SECONDS,
)
from core import chat_threads, image_store, message_media
from core.budget import Budget
from core.burst import Burst, BurstCollector
from core.context import fetch_turn_gap
from core.conversations import (
    add_participants,
    author_info,
    cap_turns,
    conversation_participant_ids,
    find_recent_conversation,
    make_turn,
    message_participant_infos,
    turn_message_ids,
)
from core.delivery import Outgoing, choose_style, deliver, render
from core.discord_text import strip_bot_mention
from core.profiles import CHAT, SPARK, Profile, profile_for
from core.prompts import THREAD_TITLE_PROMPT
from core.runner import run_turn
from core.tool_loop import EMPTY_ANSWER_FALLBACK
from core.turn import FOLLOW_HEADER, NAME_HEADER, UNPROMPTED_HEADER, TurnContext, TurnResult, is_silent

logger = logging.getLogger(__name__)

_DIRECTED = ('reply', 'mention')
# Words too common to count as the bot's name ("o bot caiu?" is not a call).
_GENERIC_NAMES = frozenset({'bot', 'robo', 'robot', 'assistente'})
_SPARK_REPORT_CACHE_SIZE = 32
# A burst of follow-ups sent while the bot answered continues that answer's
# conversation if it comes within this many seconds.
_FOLLOW_UP_SECONDS = 600
# Naming a thread holds the answer back; past this the first question names it.
_THREAD_TITLE_TIMEOUT = 8.0
_TITLE_LABEL_RE = re.compile(r'^\W*(?:t[íi]tulo|title)\W*:\s*', re.IGNORECASE)
RATE_LIMIT_TEXT = '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
ERROR_TEXT = 'Ocorreu um erro ao processar sua pergunta. Tente novamente.'


def _stays_in(conv: dict, channel: Any) -> bool:
    """Whether *conv* still talks in *channel*: it hasn't moved to a thread elsewhere."""
    thread_id = ((conv or {}).get('data') or {}).get('thread_id')
    return not thread_id or str(thread_id) == str(getattr(channel, 'id', ''))


def _turns_in_a_row(history: list[dict], author: Any) -> int:
    """How many turns in a row *author* has had with the bot, counting the one being answered."""
    author_id = str(getattr(author, 'id', ''))
    run = 1
    for turn in reversed(history):
        if str((turn.get('author') or {}).get('id', '')) != author_id:
            break
        run += 1
    return run


def _norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', text or '')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(text.lower().split())


@dataclass
class _Prepared:
    """A burst turned into the text and images of one turn."""

    question: str
    topic: str
    image_urls: list[str] = field(default_factory=list)
    ref_context: str = ''
    raw_text: str = ''


class ChatEngine:
    def __init__(self, bot: Any, services: Any, *, burst_delay: float = CHAT_BURST_SECONDS, budget: Budget | None = None):
        self.bot = bot
        self.services = services
        self.store = services.conversations
        self.bursts = BurstCollector(burst_delay, self.answer_burst, delay_for=self.burst_delay)
        self.budget = budget or Budget(
            user_limit=CHAT_USER_BUDGET, user_window=CHAT_USER_BUDGET_WINDOW,
            guild_limit=CHAT_GUILD_BUDGET, guild_window=CHAT_GUILD_BUDGET_WINDOW,
        )
        # Spark reports are too heavy to persist: kept in memory per conversation.
        self.spark_reports: TTLCache = TTLCache(maxsize=_SPARK_REPORT_CACHE_SIZE, ttl=CONVERSATIONS_TTL_SECONDS)
        # (channel, person) -> (bot message that answered them last, when): messages
        # sent while the bot was answering are older than its answer, so the
        # channel scan can't see that conversation — this can.
        self._last_answer: TTLCache = TTLCache(maxsize=2048, ttl=_FOLLOW_UP_SECONDS)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def bot_user(self) -> Any:
        return getattr(self.bot, 'user', None)

    def burst_delay(self, burst: Burst) -> float:
        """How long to wait for more messages: longer while the burst only points at something.

        "ai oh @bot" or "olha isso" with no image, link or reply yet usually has
        the thing itself coming in the next message, which may take a few
        seconds to upload.
        """
        base = self.bursts.delay
        pointing = all(
            len(self._clean(m)) <= 20
            and not message_media.visual_sources(m)
            and not getattr(getattr(m, 'reference', None), 'message_id', None)
            and not re.search(r'https?://', m.content or '')
            for m in burst.messages
        )
        return max(base, CHAT_BURST_SHORT_SECONDS) if pointing else base

    def settings_for(self, guild: Any, channel: Any) -> dict[str, Any]:
        return self.services.settings.for_channel(guild, channel)

    def bot_names(self, guild: Any) -> set[str]:
        """Normalized names the bot answers to without an @ (3+ letters each)."""
        user = self.bot_user
        me = getattr(guild, 'me', None)
        raw = [
            getattr(me, 'display_name', None), getattr(user, 'display_name', None),
            getattr(user, 'name', None), *CHAT_NAME_ALIASES,
        ]
        if guild is not None:
            raw.extend(self.services.settings.get(guild.id, None, 'aliases') or [])
        names: set[str] = set()
        for name in raw:
            norm = _norm(name or '')
            for candidate in (norm, re.sub(r'[\s_-]*bot$', '', norm).strip()):
                if len(candidate) >= 3 and candidate not in _GENERIC_NAMES:
                    names.add(candidate)
        return names

    def names_bot(self, message: Any) -> bool:
        """Whether *message* calls the bot by name (outside code, links and @mentions)."""
        text = re.sub(r'```.*?```|`[^`]*`|https?://\S+|<[@#:a][^>]*>', ' ', message.content or '', flags=re.S)
        norm = _norm(text)
        return any(
            re.search(rf'(?<![\w@./-]){re.escape(name)}(?![\w-])', norm)
            for name in self.bot_names(message.guild)
        )

    async def _react(self, message: Any, emoji: str) -> None:
        try:
            await message.add_reaction(emoji)
        except Exception:
            logger.debug('could not react on %s', getattr(message, 'id', '?'), exc_info=True)

    async def _reply_text(self, message: Any, text: str, **kwargs: Any) -> None:
        try:
            await message.reply(text, **kwargs)
        except discord.HTTPException:
            logger.debug('reply failed', exc_info=True)

    # ------------------------------------------------------------------
    # Is this message for the bot?
    # ------------------------------------------------------------------

    async def trigger_for(self, message: Any, settings: dict[str, Any]) -> str | None:
        """Why *message* reaches the bot ('reply', 'mention', 'thread', 'name') or ``None``."""
        if settings.get('respond') == 'off':
            return None
        ref = getattr(message, 'reference', None)
        ref_id = getattr(ref, 'message_id', None)
        if ref_id and await self.store.get_by_handle(ref_id):
            return 'reply'
        user = self.bot_user
        if not CHAT_MENTION_ENABLED or user is None:
            return None
        if user.mentioned_in(message):
            return 'mention'
        if settings.get('respond') != 'on':
            return None
        if chat_threads.answers_unprompted(message, user.id):
            return 'thread'
        if settings.get('name_trigger') and message.content and self.names_bot(message):
            return 'name'
        return None

    async def handle_message(self, message: Any) -> bool:
        """Router handler: claim *message* when it is for the bot (or joins an open burst)."""
        if message.author.bot or self.bot_user is None:
            return False
        settings = self.settings_for(message.guild, message.channel)
        trigger = await self.trigger_for(message, settings)
        if trigger is None:
            if settings.get('respond') == 'off':
                return False
            return await self.bursts.add(message, None)
        if not self.services.llm.available:
            if trigger in _DIRECTED:
                await self._reply_text(message, '⚠️ Chat indisponível: chave de API não configurada.', mention_author=False)
            return True
        return await self.bursts.add(message, trigger)

    # ------------------------------------------------------------------
    # Turning a burst into a question
    # ------------------------------------------------------------------

    async def reply_context(self, message: Any) -> tuple[str, list]:
        """Text block and visual sources of the message *message* replies to ('' / [] when none)."""
        ref = getattr(message, 'reference', None)
        if not ref or not getattr(ref, 'message_id', None):
            return '', []
        ref_msg = getattr(ref, 'resolved', None)
        if ref_msg is None or isinstance(ref_msg, discord.DeletedReferencedMessage):
            try:
                ref_msg = await message.channel.fetch_message(ref.message_id)
            except Exception:
                return '', []
        try:
            author = getattr(ref_msg.author, 'display_name', str(ref_msg.author))
            content = message_media.emoji_text(ref_msg.content or '').strip()
            if len(content) > 1500:
                content = content[:1500] + '…'
            images = message_media.visual_sources(ref_msg)
            media = message_media.visual_markers(ref_msg)
            files = [
                a.filename for a in getattr(ref_msg, 'attachments', [])
                if not (a.content_type and a.content_type.startswith('image/'))
            ]
            parts = [f'[Mensagem respondida — {author} (@{ref_msg.author.name})]']
            if content:
                parts.append(f'Conteúdo: {content}')
            if files:
                parts.append(f"Anexos: {', '.join(files[:5])}")
            image_atts = message_media.image_attachments(ref_msg)
            if image_atts and not content and not files:
                parts.append(f'Anexos: {len(image_atts)} imagem(ns)')
            if media:
                parts.append(f'Mídia: {media}')
            if not content and not ref_msg.attachments and not media and ref_msg.embeds:
                embed = ref_msg.embeds[0]
                text = (embed.description or embed.title or '')[:500]
                if text:
                    parts.append(f'Embed: {text}')
            return ('\n'.join(parts) if len(parts) > 1 else ''), images
        except Exception:
            logger.exception('Failed to build reply context')
            return '', []

    def _clean(self, message: Any) -> str:
        text = message.content or ''
        if self.bot_user is not None:
            text = strip_bot_mention(text, self.bot_user, message.guild)
        return re.sub(r'\s+', ' ', text).strip()

    async def prepare(self, messages: list[Any], *, skip_ref: str | int | None = None) -> _Prepared:
        """The question text and images of a burst (*skip_ref*: the conversation handle replied to)."""
        texts: list[str] = []
        images: list[str] = []
        ref_context = ''
        for m in messages:
            text = message_media.question_with_media(self._clean(m), m)
            if text:
                texts.append(text)
            images.extend(await image_store.persist_shown(message_media.visual_sources(m)))
            ref = getattr(m, 'reference', None)
            ref_id = getattr(ref, 'message_id', None)
            if not ref_context and ref_id and str(ref_id) != str(skip_ref or ''):
                ref_context, ref_images = await self.reply_context(m)
                images.extend(await image_store.persist_shown(ref_images))
        body = '\n'.join(texts)
        topic = body
        if ref_context:
            body = f"{ref_context}\n---\n{body or 'Analise a mensagem e imagem(ns) respondida(s) acima.'}"
        elif not body and images:
            body = 'Analise esta imagem.'
        raw = '\n'.join(m.content or '' for m in messages)
        return _Prepared(question=body, topic=topic or body, image_urls=images, ref_context=ref_context, raw_text=raw)

    # ------------------------------------------------------------------
    # Answering a burst
    # ------------------------------------------------------------------

    async def _find_conversation(self, burst: Burst) -> tuple[dict | None, int | None, int | None]:
        """(conversation, handle id, reply_to) the burst continues, if any.

        A reply to one of a conversation's messages continues it wherever it
        is. Being picked up by recency (the bot answered here a moment ago)
        only joins a conversation still talking in this channel: one that moved
        to a thread is over here, and pulling a new mention or name call into
        it would answer people in a thread they never joined.
        """
        channel = burst.first.channel
        for m in burst.messages:
            ref_id = getattr(getattr(m, 'reference', None), 'message_id', None)
            if ref_id:
                conv = await self.store.get_by_handle(ref_id)
                if conv:
                    return conv, ref_id, ref_id
        if all(t == 'join' for t in burst.triggers):
            handle_id = self._last_answer.get(burst.key)
            conv = await self.store.get_by_handle(handle_id) if handle_id else None
            if conv and _stays_in(conv, channel):
                return conv, handle_id, None
        directed_mention = 'mention' in burst.triggers
        if directed_mention and not CHAT_MENTION_CONTINUE_ENABLED:
            return None, None, None
        recent = await find_recent_conversation(
            self.store, channel, self.bot_user.id,
            before=burst.first, limit=CHAT_MENTION_CONTINUE_WINDOW,
            accept=lambda conv: _stays_in(conv, channel),
        )
        if recent:
            handle_id, conv = recent
            ref_id = getattr(getattr(burst.first, 'reference', None), 'message_id', None)
            return conv, handle_id, ref_id
        return None, None, None

    async def answer_burst(self, burst: Burst) -> None:
        """Answer the messages of *burst* as one turn (BurstCollector callback)."""
        directed = any(t in _DIRECTED for t in burst.triggers)
        last = burst.last
        if not self.budget.try_spend(last.author.id, getattr(last.guild, 'id', None)):
            logger.info('[chat] budget used up for user=%s guild=%s; not answering', last.author.id, getattr(last.guild, 'id', None))
            if directed:
                await self._react(last, '⏳')
            return
        conv, handle_id, reply_to = await self._find_conversation(burst)
        if conv is None:
            await self._respond(burst.messages, burst.triggers, directed=directed)
            return
        async with self.store.conversation_lock(conv['conv_id']):
            fresh = await self.store.get_by_handle(handle_id)
            await self._respond(
                burst.messages, burst.triggers, directed=directed,
                conv=fresh or conv, reply_to=reply_to, skip_ref=handle_id,
            )

    def _header(self, triggers: list[str], directed: bool) -> str | None:
        if directed:
            return None
        if 'name' in triggers:
            return NAME_HEADER
        if 'thread' in triggers:
            return UNPROMPTED_HEADER
        return FOLLOW_HEADER

    async def _respond(
        self,
        messages: list[Any],
        triggers: list[str],
        *,
        directed: bool,
        conv: dict | None = None,
        reply_to: int | None = None,
        skip_ref: int | None = None,
        replace_last: bool = False,
    ) -> None:
        last, first = messages[-1], messages[0]
        guild, channel, author = last.guild, last.channel, last.author
        settings = self.settings_for(guild, channel)
        prepared = await self.prepare(messages, skip_ref=skip_ref)
        if not prepared.question:
            if 'mention' in triggers:
                name = getattr(self.bot_user, 'display_name', 'bot')
                await self._reply_text(
                    last,
                    f'Olá {author.mention}! Me mencione com uma pergunta. Ex: @{name} como otimizar meu servidor?',
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions(users=[author]),
                )
            return
        data = (conv or {}).get('data') or {}
        history = list(data.get('turns', []))
        old_turn = history.pop() if replace_last and history else None
        header = self._header(triggers, directed)
        question = f'{header}\n---\n{prepared.question}' if header else prepared.question

        participant_infos: list[dict] = []
        for m in messages:
            add_participants(participant_infos, await message_participant_infos(m))
        participant_ids = (
            conversation_participant_ids(data, participant_infos) if conv
            else {p['id'] for p in participant_infos if p.get('id')}
        )
        gap = await fetch_turn_gap(first, history) if history else None
        report = self.spark_reports.get(conv['conv_id']) if conv else None
        profile = profile_for(data.get('profile'), has_spark_report=report is not None) if conv else CHAT
        thread_mode = settings.get('threads')
        ctx = TurnContext(
            bot=self.bot,
            profile=profile,
            question=question,
            user=author,
            guild=guild,
            channel=channel,
            created_at=last.created_at,
            message=last,
            window_before=first,
            history=history,
            image_urls=prepared.image_urls,
            context_image_urls=gap.images if gap else [],
            prior_context=gap.lines if gap else None,
            reply_to=str(reply_to) if reply_to else None,
            participant_ids=participant_ids,
            origin=last.jump_url,
            thread_blocker=(
                chat_threads.HAS_THREAD if data.get('thread_id')
                else chat_threads.thread_blocker(channel, thread_mode)
            ),
            spark_report=report if profile is SPARK else None,
            channel_settings=settings,
        )
        try:
            if directed:
                async with channel.typing():
                    result = await run_turn(ctx)
            else:
                result = await run_turn(ctx)
        except RateLimitError:
            if directed:
                await self._reply_text(last, RATE_LIMIT_TEXT)
            return
        except Exception:
            logger.exception('[chat] turn failed for message %s', last.id)
            if directed:
                await self._reply_text(last, ERROR_TEXT)
            return
        if is_silent(result.answer, unprompted=not directed):
            logger.info('[chat] stayed silent on message %s (triggers=%s)', last.id, triggers)
            return

        thread = await self._answer_thread(
            last, data, history=history, directed=directed, title=result.thread_title,
            topic=prepared.topic, answer=result.answer, mode=thread_mode,
        )
        out = self._outgoing(result, profile, settings, question=prepared.topic)
        try:
            sent = await self._send(
                out, last, thread=thread, replace=old_turn,
                mention_author=None if conv else False,
            )
        except discord.HTTPException:
            logger.exception('[chat] could not send the answer to message %s', last.id)
            return
        turn = self._make_turn(
            question, result, messages, sent, prior_context=gap.lines if gap else None,
            reply_to=reply_to, raw_text=prepared.raw_text, thread=thread,
        )
        if sent:
            self._last_answer[BurstCollector.key_for(last)] = sent[-1].id
        if conv:
            if thread is not None:
                data['thread_id'] = str(thread.id)
            data['turns'] = cap_turns(history + [turn], CONVERSATIONS_MAX_TURNS)
            add_participants(data.setdefault('participants', []), participant_infos)
            await self.store.update(
                conv['conv_id'], data,
                handle_ids=[m.id for m in sent],
                trigger_ids=[m.id for m in messages],
                remove_ids=(old_turn or {}).get('reply_ids') if replace_last else None,
            )
            if report is not None:
                self.spark_reports[conv['conv_id']] = report
        else:
            await self._create(
                sent, turn, user=author, guild=guild, channel=channel, profile=profile,
                participant_infos=participant_infos, thread=thread,
                trigger_ids=[m.id for m in messages],
            )

    def _outgoing(self, result: TurnResult, profile: Profile, settings: dict[str, Any], *, question: str = '') -> Outgoing:
        style = choose_style(result.answer, preference=settings.get('reply_style', 'auto'), structured=result.structured)
        return Outgoing(
            answer=result.answer,
            sources=result.sources,
            style=style,
            title=f'❓ {question}'[:256] if profile.question_title and question else None,
            color=profile.color,
            footer=profile.footer,
            sources_title=profile.sources_title,
        )

    async def _send(
        self, out: Outgoing, message: Any, *, thread: Any = None, replace: dict | None = None,
        mention_author: bool | None = None,
    ) -> list[Any]:
        """Send *out* for *message*; with *replace* (an edited turn), update the old answer."""
        guild = message.guild
        if replace is not None:
            edited = await self._edit_previous(out, replace, guild)
            if edited:
                return edited
        if thread is not None:
            return await deliver(out, guild=guild, thread=thread, asker=message.author)
        return await deliver(out, guild=guild, reply_to=message, mention_author=mention_author)

    async def _edit_previous(self, out: Outgoing, turn: dict, guild: Any) -> list[Any] | None:
        """Edit a single old answer in place, or delete the old answer messages (then ``None``)."""
        reply_ids = turn.get('reply_ids') or []
        channel_id = turn.get('reply_channel_id') or turn.get('channel_id')
        if not reply_ids or not channel_id:
            return None
        channel = self.bot.get_partial_messageable(int(channel_id), guild_id=getattr(guild, 'id', None))
        rendered = render(out, guild)
        if len(reply_ids) == 1 and len(rendered) == 1:
            kwargs = rendered[0]
            kwargs.setdefault('content', None)
            kwargs.setdefault('embed', None)
            kwargs.setdefault('view', None)
            try:
                return [await channel.get_partial_message(int(reply_ids[0])).edit(**kwargs)]
            except discord.HTTPException:
                logger.info('could not edit answer %s; sending a new one', reply_ids[0], exc_info=True)
        await self._delete_messages(channel, reply_ids)
        return None

    async def _delete_messages(self, channel: Any, ids: list) -> None:
        for mid in ids:
            try:
                await channel.get_partial_message(int(mid)).delete()
            except discord.HTTPException:
                logger.debug('could not delete message %s', mid, exc_info=True)

    def _make_turn(
        self, question: str, result: TurnResult, messages: list[Any], sent: list[Any], *,
        prior_context: list[str] | None, reply_to: int | None, raw_text: str, thread: Any,
    ) -> dict:
        last = messages[-1]
        turn = make_turn(
            question, result.answer,
            author=author_info(last.author),
            ts=last.created_at.timestamp(),
            message_id=last.id,
            channel_id=last.channel.id,
            channel_name=getattr(last.channel, 'name', None),
            images=result.image_urls,
            sources=result.sources,
            reply_to=reply_to,
            prior_context=prior_context,
            user_message=result.user_message,
            trajectory=result.trajectory,
            context_images=result.context_image_urls,
            message_ids=[m.id for m in messages],
            reply_ids=[m.id for m in sent],
            trigger_text=raw_text,
        )
        turn['answered_ts'] = time.time()
        if thread is not None:
            turn['reply_channel_id'] = str(thread.id)
        return turn

    async def _create(
        self, sent: list[Any], turn: dict, *, user: Any, guild: Any, channel: Any, profile: Profile,
        participant_infos: list[dict] | None = None, thread: Any = None, trigger_ids: list | None = None,
    ) -> str:
        """Store a brand-new conversation anchored on the first answer message."""
        author = author_info(user)
        origin = {
            'channel_id': str(getattr(channel, 'id', '') or ''),
            'channel_name': getattr(channel, 'name', '') or '',
            'guild_id': str(getattr(guild, 'id', '') or ''),
            'guild_name': getattr(guild, 'name', '') or '',
        }
        participants = [author] if author.get('id') else []
        add_participants(participants, participant_infos or [])
        data: dict[str, Any] = {
            'turns': [turn],
            'participants': participants,
            'origin': origin,
            'started_ts': turn.get('ts') or time.time(),
            'profile': profile.name,
        }
        if thread is not None:
            data['thread_id'] = str(thread.id)
        conv_id = str(sent[0].id)
        await self.store.create(
            conv_id,
            guild_id=origin['guild_id'] or None,
            channel_id=origin['channel_id'] or None,
            data=data,
            handle_ids=[m.id for m in sent],
            trigger_ids=trigger_ids,
        )
        return conv_id

    async def _answer_thread(
        self, message: Any, data: dict, *, history: list[dict], directed: bool,
        title: str | None, topic: str, answer: str, mode: str | None,
    ) -> discord.Thread | None:
        """Thread the answer to *message* goes to, or ``None`` to reply in place.

        A conversation that already moved keeps its thread. Otherwise one is
        opened on *message* when the model asked for it (*title*), the channel
        always uses threads, or *message*'s author has been going back and
        forth with the bot for ``CHAT_THREAD_AFTER_TURNS`` turns in a row and
        this one is addressed to it (a reply or mention). Being named in
        passing, or a group chatting with the bot, moves nobody: that is the
        channel's own conversation. The thread is named *title* (the model
        asked for it), or by the model from that back-and-forth (*topic* and
        *answer* being this turn), or after its first question.
        """
        if isinstance(message.channel, discord.Thread):
            return None
        thread = await chat_threads.conversation_thread(message.guild, data.get('thread_id'))
        if thread is not None:
            return thread
        run = _turns_in_a_row(history, message.author) if directed else 0
        if not (title or chat_threads.thread_due(message.channel, run, mode)):
            return None
        if not chat_threads.can_open_thread(message.channel, mode):
            return None
        if not title:
            earlier = history[len(history) - (run - 1):] if run > 1 else []
            first = (earlier[0].get('question', '') if earlier else '') or topic
            exchanges = [(t.get('question', ''), t.get('answer', '')) for t in earlier] + [(topic, answer)]
            title = await self.thread_name(
                exchanges, chat_threads.thread_title(first, f'Conversa com {message.author.display_name}'),
            )
        return await chat_threads.open_thread(message, title)

    async def thread_name(self, exchanges: list[tuple[str, str]], fallback: str) -> str:
        """A title for a thread the bot opens on its own, written by the model from
        the (question, answer) *exchanges* moving there; *fallback* when it can't."""
        lines = []
        for question, answer in exchanges[-4:]:
            question = (question or '').rsplit('\n---\n', 1)[-1].strip()
            lines.append(f'Pessoa: {question[:400]}\nBot: {(answer or "").strip()[:400]}')
        try:
            text = await asyncio.wait_for(
                self.services.llm.text(
                    model=CHAT_THREAD_TITLE_MODEL, system=THREAD_TITLE_PROMPT,
                    user='\n\n'.join(lines), max_tokens=200,
                ),
                timeout=_THREAD_TITLE_TIMEOUT,
            )
        except Exception:
            logger.info('[threads] could not name the thread; using the question', exc_info=True)
            return fallback
        text = next((line for line in (text or '').splitlines() if line.strip()), '')
        text = _TITLE_LABEL_RE.sub('', text.strip()).strip('"\'“”`*#.').strip()
        return chat_threads.thread_title(text, '') or fallback

    # ------------------------------------------------------------------
    # Monitors, scheduled tasks and anchored conversations
    # ------------------------------------------------------------------

    async def answer_monitor(self, message: Any, header: str) -> bool:
        """Answer *message* for the monitors that matched it, like an @mention would.

        The model may stay silent. Returns whether a reply was sent. No typing
        indicator and no error replies: nobody called the bot.
        """
        if not self.budget.try_spend(None, getattr(message.guild, 'id', None), per_user=False):
            logger.info('[monitors] server budget used up; monitor run skipped')
            return False
        prepared = await self.prepare([message])
        question = prepared.question or ('Analise esta imagem.' if prepared.image_urls else '(mensagem sem texto)')
        question = f'{header}\n---\n{question}'
        participant_infos = await message_participant_infos(message)
        settings = self.settings_for(message.guild, message.channel)
        ctx = TurnContext(
            bot=self.bot, profile=CHAT, question=question,
            user=message.author, guild=message.guild, channel=message.channel,
            created_at=message.created_at, message=message, window_before=message,
            image_urls=prepared.image_urls,
            participant_ids={p['id'] for p in participant_infos if p.get('id')},
            origin=message.jump_url, thread_blocker=chat_threads.ANSWERS_IN_PLACE,
            channel_settings=settings,
        )
        result = await run_turn(ctx)
        if is_silent(result.answer):
            return False
        out = self._outgoing(result, CHAT, settings)
        sent = await deliver(out, guild=message.guild, reply_to=message, mention_author=False)
        turn = self._make_turn(
            question, result, [message], sent, prior_context=None, reply_to=None,
            raw_text=message.content or '', thread=None,
        )
        await self._create(
            sent, turn, user=message.author, guild=message.guild, channel=message.channel,
            profile=CHAT, participant_infos=participant_infos, trigger_ids=[message.id],
        )
        return True

    async def run_task(self, prompt: str, *, guild: Any, channel: Any, owner: Any = None) -> list[Any]:
        """Run a scheduled task's prompt and post the answer in *channel*.

        Replying to the post continues it as a conversation. The model may
        answer ``[NO_REPLY]`` when the task says to post only on a condition.
        """
        now = datetime.datetime.now(BR_TZ)
        ctx = TurnContext(
            bot=self.bot, profile=CHAT, question=prompt, user=owner, guild=guild, channel=channel,
            created_at=now, thread_blocker=chat_threads.ANSWERS_IN_PLACE,
            channel_settings=self.settings_for(guild, channel),
        )
        async with channel.typing():
            result = await run_turn(ctx)
        if is_silent(result.answer, unprompted=False):
            return []
        out = self._outgoing(result, CHAT, ctx.channel_settings)
        sent = await deliver(out, guild=guild, channel=channel)
        if sent:
            turn = make_turn(
                prompt, result.answer, author=author_info(owner), ts=now.timestamp(),
                channel_id=getattr(channel, 'id', None), channel_name=getattr(channel, 'name', None),
                sources=result.sources, user_message=result.user_message, trajectory=result.trajectory,
                reply_ids=[m.id for m in sent],
            )
            await self._create(sent, turn, user=owner, guild=guild, channel=channel, profile=CHAT)
        return sent

    async def anchor_conversation(
        self, message: Any, question: str, answer: str, *, user: Any, guild: Any, channel: Any,
        created_at: datetime.datetime | None = None, profile: Profile = CHAT,
    ) -> None:
        """Start a conversation on a message the bot posted itself (e.g. a published summary)."""
        turn = make_turn(
            question, answer, author=author_info(user),
            ts=(created_at or datetime.datetime.now(datetime.timezone.utc)).timestamp(),
            channel_id=getattr(channel, 'id', None), channel_name=getattr(channel, 'name', None),
            reply_ids=[message.id],
        )
        await self._create([message], turn, user=user, guild=guild, channel=channel, profile=profile)

    # ------------------------------------------------------------------
    # Slash commands (/chat, /ask) and Spark
    # ------------------------------------------------------------------

    async def _status(self, interaction: Any, text: str | None) -> None:
        try:
            await interaction.edit_original_response(content=text)
        except discord.HTTPException:
            pass

    async def answer_interaction(
        self, interaction: Any, question: str, *, image: Any = None, profile: Profile = CHAT,
        spark_report: Any = None, title: str | None = None, deferred: bool = False,
    ) -> None:
        """Answer a slash command (``/chat``, ``/ask``) or a Spark analysis."""
        guild, channel, user = interaction.guild, interaction.channel, interaction.user
        if not deferred:
            if not self.budget.try_spend(user.id, getattr(guild, 'id', None)):
                wait = self.budget.retry_after(user.id, getattr(guild, 'id', None))
                await interaction.response.send_message(
                    f'⏳ Muitas conversas seguidas — tente de novo em {wait:.0f}s.', ephemeral=True,
                )
                return
            await interaction.response.defer(thinking=True)
            await self._status(interaction, '💭 Pensando…')
        image_url = (await image_store.persist_shown([image]) or [None])[0] if image is not None else None
        settings = self.settings_for(guild, channel)
        thread_mode = settings.get('threads')
        blocker = chat_threads.thread_blocker(channel, thread_mode)
        ctx = TurnContext(
            bot=self.bot, profile=profile, question=question, user=user, guild=guild, channel=channel,
            created_at=interaction.created_at, interaction=interaction,
            image_urls=[image_url] if image_url else [],
            thread_blocker=blocker if profile is CHAT else chat_threads.ANSWERS_IN_PLACE,
            spark_report=spark_report, can_stay_silent=False, channel_settings=settings,
            on_status=lambda text: self._status(interaction, text),
        )
        logger.info('Processing /%s user=%s guild=%s question=%r', profile.name, user.id, getattr(guild, 'id', None), question[:80])
        try:
            result = await run_turn(ctx)
        except RateLimitError:
            await self._status(interaction, None)
            await interaction.followup.send(RATE_LIMIT_TEXT)
            return
        except Exception:
            logger.exception('Error in /%s', profile.name)
            await self._status(interaction, None)
            await interaction.followup.send('Ocorreu um erro ao processar sua pergunta. Tente novamente mais tarde.')
            return
        if is_silent(result.answer, unprompted=False):
            # A slash command always needs a response: an acknowledgment, not an error.
            result.answer = '👍'
        elif not result.answer.strip():
            result.answer = EMPTY_ANSWER_FALLBACK
        out = self._outgoing(result, profile, settings, question=question)
        if title:
            out.title = title
        thread = None
        if profile is CHAT and blocker is None and (result.thread_title or chat_threads.thread_due(channel, 1, thread_mode)):
            name = result.thread_title or await self.thread_name(
                [(question, result.answer)], chat_threads.thread_title(question, f'Conversa com {user.display_name}'),
            )
            await self._status(interaction, f'🧵 Resposta na thread **{name}**')
            try:
                anchor = await interaction.original_response()
                thread = await chat_threads.open_thread(anchor, name)
            except discord.HTTPException:
                thread = None
        if thread is not None:
            sent = await deliver(out, guild=guild, thread=thread, asker=user)
        else:
            await self._status(interaction, None)
            sent = await deliver(out, guild=guild, interaction=interaction)
        if not sent:
            return
        turn = make_turn(
            question, result.answer, author=author_info(user), ts=interaction.created_at.timestamp(),
            channel_id=getattr(channel, 'id', None), channel_name=getattr(channel, 'name', None),
            images=result.image_urls, sources=result.sources, user_message=result.user_message,
            trajectory=result.trajectory, context_images=result.context_image_urls,
            reply_ids=[m.id for m in sent],
        )
        if thread is not None:
            turn['reply_channel_id'] = str(thread.id)
        conv_id = await self._create(sent, turn, user=user, guild=guild, channel=channel, profile=profile, thread=thread)
        if spark_report is not None:
            self.spark_reports[conv_id] = spark_report

    async def run_spark(self, interaction: Any, report: Any) -> None:
        """Analyse a Spark report (the interaction is already deferred) and start its conversation."""
        server_tag = f'{report.platform_brand} {report.minecraft_version}'.strip()
        title = f'🔥 Spark — {server_tag}' if server_tag else '🔥 Relatório Spark'
        await self.answer_interaction(
            interaction,
            'Analise este relatório Spark e identifique os principais problemas de desempenho, se houver. '
            'Siga o protocolo diagnóstico.',
            profile=SPARK, spark_report=report, title=title, deferred=True,
        )

    # ------------------------------------------------------------------
    # Edits and deletes of messages the bot answered
    # ------------------------------------------------------------------

    @staticmethod
    def _turn_index(data: dict, message_id: int | str) -> int | None:
        for i, turn in enumerate(data.get('turns', [])):
            if str(message_id) in turn_message_ids(turn):
                return i
        return None

    async def message_deleted(self, message_id: int, channel_id: int, guild_id: int | None) -> None:
        """A message was deleted: drop it from bursts; delete the answer it alone got."""
        self.bursts.discard(message_id)
        if not CHAT_DELETE_WITH_TRIGGER:
            return
        conv = await self.store.get_by_trigger(message_id)
        if conv is None:
            return
        async with self.store.conversation_lock(conv['conv_id']):
            conv = await self.store.get_by_trigger(message_id) or conv
            data = conv['data']
            index = self._turn_index(data, message_id)
            if index is None:
                return
            turn = data['turns'][index]
            others = [i for i in turn_message_ids(turn) if i != str(message_id)]
            if others:
                # Part of a burst: the answer still answers the rest.
                turn['message_ids'] = others
                turn['message_id'] = others[-1]
                await self.store.update(conv['conv_id'], data, remove_ids=[message_id], touch=False)
                return
            reply_ids = turn.get('reply_ids') or []
            reply_channel = turn.get('reply_channel_id') or channel_id
            channel = self.bot.get_partial_messageable(int(reply_channel), guild_id=guild_id)
            await self._delete_messages(channel, reply_ids)
            del data['turns'][index]
            logger.info('[chat] message %s deleted: removed its answer %s', message_id, reply_ids)
            if not data['turns']:
                await self.store.delete(conv['conv_id'])
            else:
                await self.store.update(
                    conv['conv_id'], data, remove_ids=[message_id, *reply_ids], touch=False,
                )

    async def message_edited(self, message_id: int, channel_id: int, guild_id: int | None, new_content: str | None) -> None:
        """A message was edited: re-answer it when it is the latest turn and the answer is recent."""
        if CHAT_EDIT_WINDOW_SECONDS <= 0 or new_content is None:
            return
        conv = await self.store.get_by_trigger(message_id)
        if conv is None or conv.get('expired'):
            return
        turns = conv['data'].get('turns', [])
        if not turns or str(message_id) not in turn_message_ids(turns[-1]):
            return
        turn = turns[-1]
        if time.time() - float(turn.get('answered_ts') or 0) > CHAT_EDIT_WINDOW_SECONDS:
            return
        channel = self.bot.get_channel(int(channel_id))
        if channel is None or not hasattr(channel, 'fetch_message'):
            return
        messages = []
        for mid in turn_message_ids(turn):
            try:
                messages.append(await channel.fetch_message(int(mid)))
            except discord.HTTPException:
                continue
        if not messages or '\n'.join(m.content or '' for m in messages) == turn.get('trigger_text'):
            return  # an embed unfurled, the text didn't change
        logger.info('[chat] message %s edited: re-answering', message_id)
        async with self.store.conversation_lock(conv['conv_id']):
            fresh = await self.store.get_by_trigger(message_id)
            if fresh is None or fresh.get('expired'):
                return
            reply_to = turn.get('reply_to')
            await self._respond(
                messages, ['reply'] * len(messages), directed=True,
                conv=fresh, reply_to=int(reply_to) if reply_to else None,
                skip_ref=reply_to, replace_last=True,
            )
