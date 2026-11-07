"""Streamed ``delta.content`` in the shapes providers actually send.

Mistral's reasoning models stream ``content`` as a list of typed chunks
(``thinking`` and ``text``) instead of a string; folding that used to raise
``TypeError: sequence item 0: expected str instance, list found``.
"""

from types import SimpleNamespace

from openai._models import construct_type
from openai.types.chat import ChatCompletionChunk

from core.llm import create_chat_completion


def _chunk(delta, finish_reason=None):
    # Built the way the SDK builds stream chunks: lenient, so a list content survives.
    return construct_type(type_=ChatCompletionChunk, value={
        'id': 'c1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'magistral-medium-latest',
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
    })


def _client(chunks):
    class _Stream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for c in chunks:
                yield c

        async def close(self):
            pass

    class _Completions:
        async def create(self, **kwargs):
            return _Stream()

    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


async def test_mistral_list_content_keeps_text_and_drops_thinking():
    client = _client([
        _chunk({'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': [{'type': 'text', 'text': 'hmm, '}]},
        ]}),
        _chunk({'content': [{'type': 'thinking', 'thinking': [{'type': 'text', 'text': 'ok'}]}]}),
        _chunk({'content': [{'type': 'text', 'text': 'Olá'}]}),
        _chunk({'content': ', mundo'}),
        _chunk({'content': [{'type': 'text', 'text': '!'}]}, 'stop'),
    ])
    response = await create_chat_completion(client, model='magistral-medium-latest', messages=[])
    msg = response.choices[0].message
    assert msg.content == 'Olá, mundo!'
    assert response.choices[0].finish_reason == 'stop'


async def test_thinking_only_stream_has_no_content():
    client = _client([
        _chunk({'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': [{'type': 'text', 'text': 'pensando'}]},
        ]}, 'stop'),
    ])
    response = await create_chat_completion(client, model='magistral-medium-latest', messages=[])
    assert response.choices[0].message.content is None
