"""Tests for channel summaries (cogs/summary.py): parsing, fetching, map-reduce, tool."""

import asyncio
import datetime
from types import SimpleNamespace

from cogs import summary
from cogs.summary import (
    LAST_MESSAGE,
    PeriodError,
    Summarizer,
    collect_messages,
    defuse_mentions,
    parse_period,
    render_citations,
    split_segments,
)
from cogs.utils import BR_TZ

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 9, 24, 15, 0, tzinfo=UTC)  # 12:00 BRT
BASE_ID = 1_300_000_000_000_000_000


def fake_message(n, author_id, content, *, minutes_ago, system=False, mentions=(), reply_to_author=None):
    reference = None
    if reply_to_author is not None:
        reference = SimpleNamespace(
            message_id=BASE_ID + 999,
            resolved=SimpleNamespace(author=SimpleNamespace(id=reply_to_author, display_name=f'u{reply_to_author}')),
        )
    return SimpleNamespace(
        id=BASE_ID + n,
        content=content,
        attachments=[],
        embeds=[],
        mentions=[SimpleNamespace(id=m) for m in mentions],
        reference=reference,
        created_at=NOW - datetime.timedelta(minutes=minutes_ago),
        author=SimpleNamespace(id=author_id, display_name=f'u{author_id}', name=f'user{author_id}', bot=False),
        jump_url=f'https://discord.com/channels/1/2/{BASE_ID + n}',
        is_system=lambda system=system: system,
    )


class FakeChannel:
    def __init__(self, messages, guild=None):
        self.id = 2
        self.name = 'geral'
        self.guild = guild
        # Newest first, like channel.history() without oldest_first.
        self._messages = sorted(messages, key=lambda m: m.created_at, reverse=True)

    def history(self, *, limit=None, before=None):
        if before is None:
            cutoff = None
        elif isinstance(before, datetime.datetime):
            cutoff = before
        else:
            cutoff = before.created_at

        async def _iterate():
            for msg in self._messages:
                if cutoff is not None and msg.created_at >= cutoff:
                    continue
                yield msg

        return _iterate()


class FakeGuild:
    def __init__(self, members=None, roles=None, channels=None):
        self._members = members or {}
        self._roles = roles or {}
        self._channels = channels or {}
        self.me = None

    def get_member(self, uid):
        return self._members.get(uid)

    def get_role(self, rid):
        return self._roles.get(rid)

    def get_channel_or_thread(self, cid):
        return self._channels.get(cid)


class FakeClient:
    """Stand-in for AsyncOpenAI: answers map calls with notes, reduce calls with a summary."""

    def __init__(self, reduce_answer):
        self.calls: list[list[dict]] = []
        self.reduce_answer = reduce_answer
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, model, messages, max_tokens):
        self.calls.append(messages)
        if messages[0]['content'].startswith(summary.MAP_SYSTEM_PROMPT[:40]):
            content = f'- notas do trecho ({len(messages[1]["content"])} chars)'
        else:
            content = self.reduce_answer
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason='stop')],
            usage=None,
        )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------


def test_parse_period():
    assert parse_period(None, now=NOW) == LAST_MESSAGE
    assert parse_period('', now=NOW) == LAST_MESSAGE
    assert parse_period('last_message', now=NOW) == LAST_MESSAGE
    assert parse_period('6h', now=NOW) == NOW - datetime.timedelta(hours=6)
    assert parse_period('30m', now=NOW) == NOW - datetime.timedelta(minutes=30)
    assert parse_period('2d', now=NOW) == NOW - datetime.timedelta(days=2)
    assert parse_period('hoje', now=NOW) == datetime.datetime(2026, 9, 24, 0, 0, tzinfo=BR_TZ)
    assert parse_period('ontem', now=NOW) == datetime.datetime(2026, 9, 23, 0, 0, tzinfo=BR_TZ)
    # Naive datetimes are Brasília time; explicit offsets are kept.
    assert parse_period('2026-09-24T08:00', now=NOW) == datetime.datetime(2026, 9, 24, 11, 0, tzinfo=UTC)
    assert parse_period('2026-09-24T08:00Z', now=NOW) == datetime.datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    assert parse_period('24/09/2026 08:00', now=NOW) == datetime.datetime(2026, 9, 24, 11, 0, tzinfo=UTC)
    try:
        parse_period('semana passada', now=NOW)
    except PeriodError:
        pass
    else:
        raise AssertionError('expected PeriodError')


def test_split_segments_by_size_and_gap():
    t0 = NOW
    step = datetime.timedelta(minutes=1)
    items = [(t0 + i * step, 'x' * 9) for i in range(10)]  # 10 chars each incl. newline
    segs = split_segments(items, max_chars=35)
    assert [len(s) for s in segs] == [3, 3, 3, 1], segs

    # A 20-minute silence closes a segment that is already half full.
    gap_items = items[:2] + [(t0 + datetime.timedelta(minutes=40), 'y' * 9)] + items[2:4]
    segs = split_segments(gap_items, max_chars=40)
    assert segs[0] == ['x' * 9, 'x' * 9], segs

    assert split_segments([], max_chars=10) == []


def test_render_citations():
    urls = {str(BASE_ID + 1): 'https://j/1', str(BASE_ID + 2): 'https://j/2'}
    text = (
        f'- A [msg_id={BASE_ID + 1}]\n'
        f'- B [msg_id={BASE_ID + 1}, msg_id={BASE_ID + 2}]\n'
        f'- C [msg_id=1999999999999999999]\n'
        f'- D (msg_id={BASE_ID + 2})'
    )
    out = render_citations(text, urls)
    assert out.splitlines() == [
        '- A [↗](https://j/1)',
        '- B [↗](https://j/1) [↗](https://j/2)',
        '- C',
        '- D ([↗](https://j/2))',
    ], out


def test_defuse_mentions():
    guild = FakeGuild(
        members={111111111111111111: SimpleNamespace(display_name='Ana')},
        roles={222222222222222222: SimpleNamespace(name='Staff')},
    )
    text = 'oi <@111111111111111111> e <@!333333333333333333>, <@&222222222222222222>, @joao, @everyone, a@b.com'
    out = defuse_mentions(text, guild)
    assert out == 'oi Ana e alguém, Staff, joao, everyone, a@b.com', out


def test_collect_since_last_message():
    requester = 7
    msgs = [
        fake_message(1, 5, 'antes de tudo', minutes_ago=300),
        fake_message(2, requester, 'minha última antes de sair', minutes_ago=200),
        fake_message(3, 5, 'fulano falou', minutes_ago=100),
        fake_message(4, 6, 'entrou', minutes_ago=90, system=True),
        fake_message(5, 6, 'ciclano respondeu', minutes_ago=50),
        fake_message(6, requester, 'voltei!', minutes_ago=3),  # within grace: included
        fake_message(7, requester, '@bot o que perdi?', minutes_ago=0),  # trigger: excluded
    ]
    trigger = msgs[-1]
    fetched = run(collect_messages(
        FakeChannel(msgs), requester_id=requester, since=LAST_MESSAGE,
        before=trigger, now=NOW,
    ))
    assert [m.id - BASE_ID for m in fetched.messages] == [3, 5, 6]
    assert fetched.anchor_found and fetched.since == msgs[1].created_at
    assert not fetched.hit_cap


def test_collect_cap_and_max_days():
    msgs = [fake_message(i, 5, f'm{i}', minutes_ago=100 - i) for i in range(10)]
    fetched = run(collect_messages(
        FakeChannel(msgs), requester_id=7, since=NOW - datetime.timedelta(days=1),
        now=NOW, max_messages=4,
    ))
    assert [m.id - BASE_ID for m in fetched.messages] == [6, 7, 8, 9]  # newest 4, chronological
    assert fetched.hit_cap and fetched.since == msgs[6].created_at

    old = [fake_message(1, 5, 'velha', minutes_ago=60 * 24 * 10), fake_message(2, 5, 'nova', minutes_ago=10)]
    fetched = run(collect_messages(
        FakeChannel(old), requester_id=7, since=NOW - datetime.timedelta(days=30),
        now=NOW, max_days=7,
    ))
    assert [m.id - BASE_ID for m in fetched.messages] == [2]
    assert fetched.hit_max_days


def test_summarize_single_call_renders_links_and_mentions():
    requester = SimpleNamespace(id=7, display_name='Eu')
    msgs = [
        fake_message(1, requester.id, 'saindo', minutes_ago=120),
        fake_message(2, 5, 'o servidor caiu', minutes_ago=60),
        fake_message(3, 6, 'reiniciei, voltou', minutes_ago=50, reply_to_author=5),
        fake_message(4, 6, 'Eu, dá uma olhada depois', minutes_ago=40, mentions=(requester.id,)),
    ]
    reduce = (
        f'**Tópicos**\n- Servidor caiu, @user6 reiniciou [msg_id={BASE_ID + 2}] '
        f'[msg_id=1999999999999999999]'
    )
    client = FakeClient(reduce)
    result = run(Summarizer(client, 'm').summarize(
        FakeChannel(msgs), requester=requester, since=LAST_MESSAGE, now=NOW,
    ))
    assert len(client.calls) == 1
    prompt = client.calls[0][1]['content']
    assert f'[msg_id={BASE_ID + 2}]' in prompt and 'saindo' not in prompt
    assert result.message_count == 3 and result.segments == 1
    assert result.text == (
        f'**Tópicos**\n- Servidor caiu, user6 reiniciou [↗](https://discord.com/channels/1/2/{BASE_ID + 2})'
    ), result.text
    assert len(result.mentions) == 1 and 'dá uma olhada' in result.mentions[0]
    assert result.period_note.startswith('desde a última mensagem de Eu')


def test_summarize_map_reduce():
    msgs = [fake_message(i, 5, 'x' * 200, minutes_ago=200 - i) for i in range(20)]
    client = FakeClient('**Tópicos**\n- tudo')
    result = run(Summarizer(client, 'm', segment_chars=1500).summarize(
        FakeChannel(msgs), requester=SimpleNamespace(id=7, display_name='Eu'),
        since=NOW - datetime.timedelta(hours=6), now=NOW,
    ))
    map_calls = [c for c in client.calls if c[0]['content'].startswith(summary.MAP_SYSTEM_PROMPT[:40])]
    assert result.segments > 1 and len(map_calls) == result.segments
    assert len(client.calls) == result.segments + 1
    assert 'notas do trecho' in client.calls[-1][1]['content']  # reduce sees the map notes
    assert result.text == '**Tópicos**\n- tudo'


def test_summarize_nothing_new_skips_llm():
    requester = SimpleNamespace(id=7, display_name='Eu')
    msgs = [fake_message(1, requester.id, 'só eu aqui', minutes_ago=5)]
    client = FakeClient('não deveria ser chamado')
    result = run(Summarizer(client, 'm').summarize(
        FakeChannel(msgs), requester=requester, since=NOW - datetime.timedelta(hours=6), now=NOW,
    ))
    assert result.empty and client.calls == []


def _on_real_clock(msgs):
    """Shift fixture messages so they sit the same distance from the real
    clock as from NOW; exec_tool and /resumo count periods back from now()."""
    shift = datetime.datetime.now(UTC) - NOW
    for m in msgs:
        m.created_at += shift
    return msgs


def test_exec_tool_permissions_and_output():
    guild = FakeGuild()
    other_guild = FakeGuild()
    msgs = _on_real_clock([fake_message(1, 5, 'deploy feito', minutes_ago=30)])
    channel = FakeChannel(msgs, guild=guild)
    foreign = FakeChannel([], guild=other_guild)
    foreign.id = 444444444444444444
    guild._channels = {channel.id: channel, foreign.id: foreign}
    cog = summary.ChannelSummary(SimpleNamespace())
    cog.summarizer = Summarizer(FakeClient(f'**Tópicos**\n- Deploy [msg_id={BASE_ID + 1}]'), 'm')

    # No member behind the request (e.g. a scheduled job whose creator left).
    text, _ = run(cog.exec_tool({}, guild=guild, channel=channel, requester=None))
    assert 'não tem acesso' in text, text

    text, _ = run(cog.exec_tool({'channel_id': '<#444444444444444444>'}, guild=guild, channel=channel, requester=None))
    assert 'deste servidor' in text, text

    original = summary.can_read
    try:
        async def _allow(member, ch):
            return True
        summary.can_read = _allow
        requester = SimpleNamespace(id=7, display_name='Eu')
        text, _ = run(cog.exec_tool({'since': 'agora mesmo'}, guild=guild, channel=channel, requester=requester))
        assert 'Período inválido' in text, text
        text, _ = run(cog.exec_tool({'since': '2h'}, guild=guild, channel=channel, requester=requester))
        assert 'channel_id=2' in text and '1 mensagens' in text, text
        assert f'[↗](https://discord.com/channels/1/2/{BASE_ID + 1})' in text, text
    finally:
        summary.can_read = original


class FakeResponse:
    def __init__(self, sink):
        self.sink = sink

    async def send_message(self, content=None, **kwargs):
        self.sink.append(('send_message', content, kwargs))

    async def defer(self, **kwargs):
        self.sink.append(('defer', None, kwargs))


class FakeInteraction:
    def __init__(self, *, user, guild, channel):
        self.events: list[tuple] = []
        self.user = user
        self.guild = guild
        self.guild_id = 1
        self.channel = channel
        self.created_at = NOW
        self.response = FakeResponse(self.events)

    async def edit_original_response(self, **kwargs):
        self.events.append(('edit', None, kwargs))

    async def original_response(self):
        return SimpleNamespace(id=BASE_ID + 500)


def test_resumo_command_and_publish():
    guild = FakeGuild()
    msgs = _on_real_clock([
        fake_message(1, 5, 'build nova no ar', minutes_ago=30),
        fake_message(2, 6, 'ok, <@7> testa aí', minutes_ago=20, mentions=(7,)),
    ])
    channel = FakeChannel(msgs, guild=guild)
    stored = []

    class FakeCommandsCog:
        async def _store_new_conversation(self, msg, question, answer, sources, **kwargs):
            stored.append((msg.id, question, answer))

    bot = SimpleNamespace(get_cog=lambda name: FakeCommandsCog() if name == 'Commands' else None)
    cog = summary.ChannelSummary(bot)
    cog.summarizer = Summarizer(FakeClient(f'**Tópicos**\n- Build nova [msg_id={BASE_ID + 1}]'), 'm')
    user = SimpleNamespace(id=7, display_name='Eu')

    original = summary.can_read
    try:
        async def _allow(member, ch):
            return True
        summary.can_read = _allow

        async def _flow():
            itx = FakeInteraction(user=user, guild=guild, channel=channel)
            await cog.resumo.callback(cog, itx, None, 'banana', None)
            assert itx.events[0][0] == 'send_message' and 'Período inválido' in itx.events[0][1]

            itx = FakeInteraction(user=user, guild=guild, channel=channel)
            await cog.resumo.callback(cog, itx, None, '6h', None)
            assert itx.events[0] == ('defer', None, {'ephemeral': True, 'thinking': True}), itx.events[0]
            final = itx.events[-1][2]
            embed, view = final['embed'], final['view']
            assert embed.title == '📝 Resumo de #geral'
            assert f'[↗](https://discord.com/channels/1/2/{BASE_ID + 1})' in embed.description
            assert [f.name for f in embed.fields] == ['🔔 Mencionaram você']
            assert embed.footer.text.startswith('2 mensagens · desde ')

            click = FakeInteraction(user=user, guild=guild, channel=channel)
            await view.publish.callback(click)
            kind, _, kwargs = click.events[0]
            public = kwargs['embed']
            assert kind == 'send_message' and not public.fields  # mentions stay private
            assert public.footer.text.endswith('pedido por Eu')
            assert ('edit', None, {'view': None}) in itx.events  # button removed from the ephemeral
            assert stored and stored[0][0] == BASE_ID + 500 and stored[0][1].startswith('/resumo #geral')

        asyncio.run(_flow())
    finally:
        summary.can_read = original


if __name__ == '__main__':
    test_parse_period()
    test_split_segments_by_size_and_gap()
    test_render_citations()
    test_defuse_mentions()
    test_collect_since_last_message()
    test_collect_cap_and_max_days()
    test_summarize_single_call_renders_links_and_mentions()
    test_summarize_map_reduce()
    test_summarize_nothing_new_skips_llm()
    test_exec_tool_permissions_and_output()
    test_resumo_command_and_publish()
    print('ok')
