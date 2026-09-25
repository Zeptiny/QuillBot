"""Regression tests for images attached to automatically fetched context."""

import asyncio
import datetime
from types import SimpleNamespace

from cogs import conversation_store as cs
from cogs import utils
from cogs.conversation_store import build_current_message


class FakeChannel:
    def __init__(self, messages):
        self.id = 99
        self.name = 'general'
        self._messages = messages

    def history(self, *, limit, before=None):
        async def _iterate():
            for message in self._messages:
                if before is not None and message.id >= before.id:
                    continue
                yield message

        return _iterate()


def fake_message(message_id, content, *, image=False, bot=False):
    attachments = []
    if image:
        attachments.append(SimpleNamespace(
            filename=f'image-{message_id}.png',
            content_type='image/png',
        ))
    return SimpleNamespace(
        id=message_id,
        content=content,
        attachments=attachments,
        embeds=[],
        author=SimpleNamespace(display_name=f'user-{message_id}', name=f'user-{message_id}', id=message_id, bot=bot),
        created_at=datetime.datetime.now(datetime.timezone.utc),
        reference=None,
    )


def test_followup_gap_preserves_images():
    messages = [
        fake_message(3, 'newer', image=True),
        fake_message(2, 'older', image=True),
    ]
    channel = FakeChannel(messages)
    current = fake_message(4, 'question')
    current.channel = channel

    original = utils.image_store.persist_images

    async def persist_images(attachments):
        return [f'img:{attachment.filename}' for attachment in list(attachments)]

    utils.image_store.persist_images = persist_images
    try:
        context = asyncio.run(utils.fetch_turn_gap(
            current,
            [{'message_id': '1', 'channel_id': str(channel.id)}],
        ))
        assert context.lines[0].endswith('older [anexo:image-2.png] [msg_id=2]')
        assert context.lines[1].endswith('newer [anexo:image-3.png] [msg_id=3]')
        assert context.images == ['img:image-2.png', 'img:image-3.png']

        recent = asyncio.run(utils.fetch_recent_channel_context(
            SimpleNamespace(), channel, before=current, limit=2,
        ))
        assert recent.text.startswith('<mensagens_recentes_do_canal>')
        assert recent.images == ['img:image-2.png', 'img:image-3.png']
    finally:
        utils.image_store.persist_images = original


def test_current_message_inlines_context_images_with_current_image_priority():
    original_is_ref = cs.image_store.is_image_ref
    original_part = cs.image_store.image_part
    try:
        cs.image_store.is_image_ref = lambda value: value.startswith('img:')
        cs.image_store.image_part = lambda ref: {
            'type': 'image_url',
            'image_url': {'url': f'data:image/jpeg;base64,{ref}'},
        }
        message = build_current_message(
            'question',
            author=None,
            ts=None,
            image_urls=['img:current.jpg'],
            context_image_urls=[
                'img:context-1.jpg', 'img:context-2.jpg',
                'img:context-3.jpg', 'img:context-4.jpg',
            ],
        )
        parts = message['content']
        assert [part['image_url']['url'] for part in parts[1:]] == [
            'data:image/jpeg;base64,img:current.jpg',
            'data:image/jpeg;base64,img:context-2.jpg',
            'data:image/jpeg;base64,img:context-3.jpg',
            'data:image/jpeg;base64,img:context-4.jpg',
        ]
        # The channel image left out of the budget is not the user's: no
        # "user shared an image" marker may be emitted for it.
        assert 'o usuário compartilhou' not in parts[0]['text']
        assert 'as 3 imagens mais recentes' in parts[0]['text']
        assert 'depois das enviadas pelo usuário' in parts[0]['text']

        overflow = build_current_message(
            'question', author=None, ts=None,
            image_urls=[f'img:current-{i}.jpg' for i in range(5)],
            context_image_urls=['img:context-1.jpg'],
        )
        text = overflow['content'][0]['text']
        assert '[o usuário compartilhou uma imagem neste turno]' in text
        assert 'imagens mais recentes anexadas' not in text
        assert len(overflow['content']) == 5
    finally:
        cs.image_store.is_image_ref = original_is_ref
        cs.image_store.image_part = original_part


def test_turn_keeps_context_images_apart():
    """Replay re-sends channel images but never labels them as the user's."""
    original_is_ref = cs.image_store.is_image_ref
    original_part = cs.image_store.image_part
    try:
        cs.image_store.is_image_ref = lambda value: value.startswith('img:')
        cs.image_store.image_part = lambda ref: {
            'type': 'image_url',
            'image_url': {'url': f'data:image/jpeg;base64,{ref}'},
        }
        current, context, dropped = cs.select_image_refs(
            ['img:user.jpg'], ['img:c1.jpg', 'img:c2.jpg'],
        )
        assert (current, context, dropped) == (['img:user.jpg'], ['img:c1.jpg', 'img:c2.jpg'], 0)
        sent = build_current_message(
            'question', author=None, ts=None,
            image_urls=current, context_image_urls=context,
        )
        turn = cs.make_turn(
            'question', 'answer',
            author={'id': '1', 'name': 'u', 'display': 'U'}, ts=1.0,
            images=current, context_images=context, user_message=sent,
        )
        assert turn['images'] == ['img:user.jpg']
        assert turn['context_images'] == ['img:c1.jpg', 'img:c2.jpg']

        full = cs.build_history_messages([turn], trajectory_turns=6, max_images=4)
        assert full[0]['content'] == sent['content']  # byte-identical replay

        tight = cs.build_history_messages([turn], trajectory_turns=6, max_images=2)
        text = tight[0]['content'][0]['text']
        urls = [p['image_url']['url'] for p in tight[0]['content'][1:]]
        assert urls == ['data:image/jpeg;base64,img:user.jpg', 'data:image/jpeg;base64,img:c2.jpg']
        assert '[uma imagem do canal deste turno não foi reenviada]' in text
        assert 'o usuário compartilhou' not in text

        compact = cs.build_history_messages([turn], trajectory_turns=0, max_images=4)
        compact_urls = [p['image_url']['url'] for p in compact[0]['content'][1:]]
        assert compact_urls == ['data:image/jpeg;base64,img:user.jpg']
    finally:
        cs.image_store.is_image_ref = original_is_ref
        cs.image_store.image_part = original_part


def test_context_images_switch_and_age_limit():
    old = fake_message(2, 'old', image=True)
    old.created_at -= datetime.timedelta(hours=3)
    messages = [fake_message(3, 'new', image=True), old]
    channel = FakeChannel(messages)
    current = fake_message(4, 'question')

    original = utils.image_store.persist_images
    original_enabled = utils.CHANNEL_CONTEXT_IMAGES_ENABLED
    original_age = utils.CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES

    async def persist_images(attachments):
        return [f'img:{attachment.filename}' for attachment in list(attachments)]

    utils.image_store.persist_images = persist_images
    try:
        utils.CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES = 60
        recent = asyncio.run(utils.fetch_recent_channel_context(
            SimpleNamespace(), channel, before=current, limit=2,
        ))
        assert recent.images == ['img:image-3.png']
        assert '[anexo:image-2.png]' in recent.text  # still listed as text

        utils.CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES = 0
        recent = asyncio.run(utils.fetch_recent_channel_context(
            SimpleNamespace(), channel, before=current, limit=2,
        ))
        assert recent.images == ['img:image-2.png', 'img:image-3.png']

        utils.CHANNEL_CONTEXT_IMAGES_ENABLED = False
        recent = asyncio.run(utils.fetch_recent_channel_context(
            SimpleNamespace(), channel, before=current, limit=2,
        ))
        assert recent.images == []
        assert len(recent.lines) == 2
    finally:
        utils.image_store.persist_images = original
        utils.CHANNEL_CONTEXT_IMAGES_ENABLED = original_enabled
        utils.CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES = original_age


def test_attachment_download_is_reused():
    import os
    import tempfile

    store = utils.image_store
    original_dir = store.IMAGES_DIR
    original_persist = store.persist
    reads = []

    class Attachment:
        id = 555
        url = 'https://cdn.example/x.png'

        async def read(self):
            reads.append(1)
            return b'bytes'

    async def persist(data):
        ref = 'img:stored.jpg'
        with open(store.image_path(ref), 'wb') as f:
            f.write(data)
        return ref

    with tempfile.TemporaryDirectory() as tmp:
        store.IMAGES_DIR = tmp
        store.persist = persist
        store._attachment_refs.clear()
        try:
            first = asyncio.run(store.persist_attachment(Attachment()))
            second = asyncio.run(store.persist_attachment(Attachment()))
            assert first == second == 'img:stored.jpg'
            assert len(reads) == 1

            os.remove(store.image_path(first))  # swept: must download again
            third = asyncio.run(store.persist_attachment(Attachment()))
            assert third == 'img:stored.jpg'
            assert len(reads) == 2
        finally:
            store.IMAGES_DIR = original_dir
            store.persist = original_persist
            store._attachment_refs.clear()


def test_mention_context_images_are_labeled():
    """A bare @mention has no own images; the channel ones must be announced."""
    original_is_ref = cs.image_store.is_image_ref
    original_part = cs.image_store.image_part
    try:
        cs.image_store.is_image_ref = lambda value: value.startswith('img:')
        cs.image_store.image_part = lambda ref: {
            'type': 'image_url',
            'image_url': {'url': f'data:image/jpeg;base64,{ref}'},
        }
        message = build_current_message(
            'o que tem na imagem acima?',
            author=None,
            ts=None,
            context_blocks='<mensagens_recentes_do_canal>…</mensagens_recentes_do_canal>',
            context_image_urls=['img:context-1.jpg'],
        )
        parts = message['content']
        assert [part['type'] for part in parts] == ['text', 'image_url']
        assert 'a imagem mais recente anexada nas mensagens do canal acima' in parts[0]['text']
        assert 'depois das enviadas' not in parts[0]['text']
        assert 'o usuário compartilhou' not in parts[0]['text']

        plain = build_current_message('pergunta', author=None, ts=None)
        assert plain['content'] == 'pergunta'
    finally:
        cs.image_store.is_image_ref = original_is_ref
        cs.image_store.image_part = original_part


if __name__ == '__main__':
    test_followup_gap_preserves_images()
    test_current_message_inlines_context_images_with_current_image_priority()
    test_mention_context_images_are_labeled()
    test_turn_keeps_context_images_apart()
    test_context_images_switch_and_age_limit()
    test_attachment_download_is_reused()
    print('ok')
