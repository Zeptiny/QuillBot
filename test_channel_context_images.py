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
            'data:image/jpeg;base64,img:context-1.jpg',
            'data:image/jpeg;base64,img:context-2.jpg',
            'data:image/jpeg;base64,img:context-3.jpg',
        ]
        assert '[o usuário compartilhou uma imagem neste turno]' in parts[0]['text']
    finally:
        cs.image_store.is_image_ref = original_is_ref
        cs.image_store.image_part = original_part


if __name__ == '__main__':
    test_followup_gap_preserves_images()
    test_current_message_inlines_context_images_with_current_image_priority()
    print('ok')
