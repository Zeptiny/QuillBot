"""Tests for stickers, GIF embeds, custom emojis and reactions in channel context."""

import asyncio
import datetime
import io
from types import SimpleNamespace

from PIL import Image

from cogs import image_store
from cogs import message_media as mm
from cogs import utils


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def proxy(url=None, proxy_url=None):
    return SimpleNamespace(url=url, proxy_url=proxy_url)


def tenor_embed():
    return SimpleNamespace(
        type='gifv',
        title=None,
        url='https://tenor.com/view/cat-dance-gif-25010353',
        thumbnail=proxy(
            'https://media.tenor.com/5ZT5y0Vw1GMAAAAe/cat-dance.png',
            'https://images-ext-1.discordapp.net/external/abc/cat-dance.png',
        ),
        video=proxy('https://media.tenor.com/5ZT5y0Vw1GMAAAPo/cat-dance.mp4'),
        image=proxy(),
    )


def sticker(name, fmt='png', sticker_id=1):
    return SimpleNamespace(id=sticker_id, name=name, format=SimpleNamespace(name=fmt), url=f'https://s/{sticker_id}')


class FakeReaction:
    def __init__(self, emoji, users, count=None, delay=0.0):
        self.emoji = emoji
        self._users = users
        self.count = len(users) if count is None else count
        self.delay = delay
        self.calls = 0

    def users(self, *, limit=None):
        self.calls += 1

        async def _iterate():
            if self.delay:
                await asyncio.sleep(self.delay)
            for name in self._users[:limit]:
                yield SimpleNamespace(display_name=name)

        return _iterate()


def fake_message(message_id, content='', *, attachments=(), stickers=(), embeds=(), reactions=(), created_at=None):
    return SimpleNamespace(
        id=message_id,
        content=content,
        attachments=list(attachments),
        stickers=list(stickers),
        embeds=list(embeds),
        reactions=list(reactions),
        author=SimpleNamespace(display_name=f'user-{message_id}', name=f'user-{message_id}', id=message_id, bot=False),
        created_at=created_at or _now(),
        reference=None,
    )


class FakeChannel:
    def __init__(self, messages):
        self.id = 99
        self.name = 'general'
        self._messages = messages

    def history(self, *, limit, before=None):
        async def _iterate():
            for message in self._messages[:limit]:
                yield message

        return _iterate()


def test_content_text_markers():
    msg = fake_message(
        1, 'olha <:kek:123456789012345678> e <a:party:223456789012345678>',
        stickers=[sticker('Pepe Feliz')],
        embeds=[tenor_embed()],
    )
    text = utils.message_content_text(msg)
    assert text == 'olha :kek: e :party: [figurinha:Pepe Feliz] [gif: cat dance]'

    only_sticker = fake_message(2, stickers=[sticker('Oi')])
    assert utils.message_content_text(only_sticker) == '[figurinha:Oi]'


def test_media_sources_order_and_lottie():
    image = SimpleNamespace(id=5, filename='a.png', content_type='image/png', url='u')
    pdf = SimpleNamespace(id=6, filename='a.pdf', content_type='application/pdf', url='u')
    msg = fake_message(
        1, '<:kek:123456789012345678>',
        attachments=[image, pdf],
        stickers=[sticker('ok', sticker_id=7), sticker('vector', fmt='lottie', sticker_id=8)],
        embeds=[tenor_embed(), SimpleNamespace(type='rich', title='x')],
    )
    sources = mm.visual_sources(msg)
    assert sources[0] is image
    assert sources[1].name == 'ok'
    assert isinstance(sources[2], mm.LinkedMedia)
    assert sources[2].urls == (
        'https://media.tenor.com/5ZT5y0Vw1GMAAAAM/cat-dance.gif',
        'https://images-ext-1.discordapp.net/external/abc/cat-dance.png',
        'https://media.tenor.com/5ZT5y0Vw1GMAAAAe/cat-dance.png',
    )
    assert isinstance(sources[3], mm.EmojiSheet)
    assert len(sources) == 4
    assert '[figurinha:vector]' in utils.message_content_text(msg)  # lottie: name only


def test_giphy_urls_and_label():
    embed = SimpleNamespace(
        type='gifv', title=None,
        url='https://giphy.com/gifs/cat-dance-3o7TKSjRrfIPjeiVyM',
        thumbnail=proxy('https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy_s.gif'),
        video=proxy(), image=proxy(),
    )
    media = mm.embed_media(embed)
    assert media.urls[0] == 'https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/200w.gif'
    assert mm.embed_markers(fake_message(1, embeds=[embed])) == ['[gif: cat dance]']


def test_linked_media_falls_back():
    original = image_store.download
    tried = []

    async def download(url):
        tried.append(url)
        return b'still' if url.endswith('.png') else None

    image_store.download = download
    try:
        data = asyncio.run(mm.LinkedMedia(['https://x/a.gif', 'https://x/b.png']).read())
        assert data == b'still'
        assert tried == ['https://x/a.gif', 'https://x/b.png']
    finally:
        image_store.download = original


def test_context_groups_media_first_then_emoji_sheets():
    img = lambda i: SimpleNamespace(id=i, filename=f'{i}.png', content_type='image/png', url='u')
    messages = [  # newest first
        fake_message(6, '<:a:111111111111111111>'),
        fake_message(5, attachments=[img(50)]),
        fake_message(4, '<:b:222222222222222222>', stickers=[sticker('s', sticker_id=40)]),
        fake_message(3, attachments=[img(30), img(31)]),
        fake_message(2, '<:c:333333333333333333>'),
    ]
    groups = utils._context_image_groups(messages, _now())
    kinds = [[type(s).__name__ for s in g] for g in groups]
    # 4 slots: 1 + 1 + 2 media, nothing left for emoji sheets
    assert kinds == [['SimpleNamespace'], ['SimpleNamespace'], ['SimpleNamespace', 'SimpleNamespace']]

    messages[3].attachments = []
    groups = utils._context_image_groups(messages, _now())
    kinds = [[type(s).__name__ for s in g] for g in groups]
    # media take 2 slots; the two newest emoji messages get sheets
    assert kinds == [['EmojiSheet'], ['SimpleNamespace'], ['SimpleNamespace', 'EmojiSheet']]


def test_reaction_summaries_names_counts_and_cache():
    mm._reactors.clear()
    thumbs = FakeReaction('👍', ['Ana', 'Bruno', 'Caio', 'Dani'], count=6)
    kek = FakeReaction(SimpleNamespace(id=9, name='kek'), ['Eva'])
    msg = fake_message(1, 'oi', reactions=[thumbs, kek])
    original_limit = mm.REACTION_USERS_LIMIT
    mm.REACTION_USERS_LIMIT = 3
    try:
        out = asyncio.run(mm.reaction_summaries([msg]))
        assert out == {1: '[reações: 👍 6 (Ana, Bruno, Caio +3); :kek: 1 (Eva)]'}
        asyncio.run(mm.reaction_summaries([msg]))
        assert thumbs.calls == 1 and kek.calls == 1  # cached while count unchanged

        thumbs.count = 7
        asyncio.run(mm.reaction_summaries([msg]))
        assert thumbs.calls == 2

        mm.REACTION_USERS_LIMIT = 0
        mm._reactors.clear()
        out = asyncio.run(mm.reaction_summaries([msg]))
        assert out == {1: '[reações: 👍 7; :kek: 1]'}
    finally:
        mm.REACTION_USERS_LIMIT = original_limit
        mm._reactors.clear()


def test_reaction_summaries_timeout_and_disabled():
    mm._reactors.clear()
    slow = FakeReaction('🔥', ['Ana'], delay=1.0)
    msg = fake_message(1, 'oi', reactions=[slow])
    original_timeout = mm.REACTION_FETCH_TIMEOUT
    original_enabled = mm.CHANNEL_CONTEXT_REACTIONS_ENABLED
    mm.REACTION_FETCH_TIMEOUT = 0.05
    try:
        out = asyncio.run(mm.reaction_summaries([msg]))
        assert out == {1: '[reações: 🔥 1]'}
        mm.CHANNEL_CONTEXT_REACTIONS_ENABLED = False
        assert asyncio.run(mm.reaction_summaries([msg])) == {}
    finally:
        mm.REACTION_FETCH_TIMEOUT = original_timeout
        mm.CHANNEL_CONTEXT_REACTIONS_ENABLED = original_enabled
        mm._reactors.clear()


def test_channel_history_lines_carry_reactions():
    mm._reactors.clear()
    messages = [
        fake_message(2, 'boa', reactions=[FakeReaction('😂', ['Ana', 'Bruno'])]),
        fake_message(1, 'sem reação'),
    ]
    text = asyncio.run(utils.fetch_channel_history(SimpleNamespace(), FakeChannel(messages), limit=5))
    lines = text.splitlines()[1:]
    assert lines[0].endswith(': sem reação [msg_id=1]')
    assert lines[1].endswith(': boa [reações: 😂 2 (Ana, Bruno)] [msg_id=2]')
    mm._reactors.clear()


def _png(img):
    out = io.BytesIO()
    img.save(out, format='PNG')
    return out.getvalue()


def test_encode_animation_sheet_and_transparency():
    frames = [Image.new('RGB', (40, 40), color) for color in ('red', 'green', 'blue', 'yellow', 'white', 'black')]
    gif = io.BytesIO()
    frames[0].save(gif, format='GIF', save_all=True, append_images=frames[1:], duration=80, loop=0)
    sheet = Image.open(io.BytesIO(image_store._encode(gif.getvalue())))
    cell = (image_store.IMAGE_MAX_SIDE - image_store._CAPTION_H) // 2
    assert sheet.size == (2 * cell, 2 * cell + image_store._CAPTION_H)  # fits unscaled
    # 4 evenly spaced frames of 6 (0, 2, 3, 5): red, blue, yellow, black
    for (cx, cy), expected in {
        (0, 0): (255, 0, 0), (1, 0): (0, 0, 255), (0, 1): (255, 255, 0), (1, 1): (0, 0, 0),
    }.items():
        pixel = sheet.getpixel((cx * cell + cell // 2, cy * cell + cell // 2))
        assert all(abs(a - b) < 40 for a, b in zip(pixel, expected)), (cx, cy, pixel)

    transparent = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
    still = Image.open(io.BytesIO(image_store._encode(_png(transparent))))
    assert min(still.getpixel((16, 16))) > 240  # white, not black

    grid = Image.open(io.BytesIO(image_store.labeled_grid([
        (_png(Image.new('RGBA', (64, 64), (255, 0, 0, 255))), ':kek:'),
        (b'not an image', ':broken:'),
    ])))
    assert grid.width == 96  # one decodable tile


def test_question_with_media():
    msg = fake_message(1, stickers=[sticker('Oi')])
    assert mm.question_with_media('e isso <:kek:123456789012345678>?', msg) == 'e isso :kek:? [figurinha:Oi]'
    assert mm.question_with_media('', fake_message(2)) == ''


if __name__ == '__main__':
    test_content_text_markers()
    test_media_sources_order_and_lottie()
    test_giphy_urls_and_label()
    test_linked_media_falls_back()
    test_context_groups_media_first_then_emoji_sheets()
    test_reaction_summaries_names_counts_and_cache()
    test_reaction_summaries_timeout_and_disabled()
    test_channel_history_lines_carry_reactions()
    test_encode_animation_sheet_and_transparency()
    test_question_with_media()
    print('ok')
