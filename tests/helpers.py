"""Assertion helper shared by the tests."""


def check(name, cond, detail=''):
    """Assert `cond`, labelling the failure with `name` and the value seen."""
    assert cond, f'{name}: {detail}' if detail != '' else name


async def as_stream(response):
    """Replay a non-streaming fake completion as streamed chunks.

    The bot always calls ``chat.completions.create(stream=True)``; fakes keep
    building whole responses and return ``as_stream(resp)`` instead. Content
    and tool-call arguments are split across chunks so reassembly is exercised.
    """
    from types import SimpleNamespace

    choice = response.choices[0]
    message = choice.message
    known = {'role', 'content', 'tool_calls'}
    extras = {k: v for k, v in vars(message).items() if k not in known and isinstance(v, str)}

    def chunk(delta=None, finish_reason=None, usage=None, choices=True):
        ch = [SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)] if choices else []
        return SimpleNamespace(id='chatcmpl-fake', model='fake', created=0, choices=ch, usage=usage)

    def delta(**fields):
        return SimpleNamespace(
            role=fields.pop('role', None), content=fields.pop('content', None),
            tool_calls=fields.pop('tool_calls', None), model_extra=fields,
        )

    yield chunk(delta(role='assistant', **extras))
    content = getattr(message, 'content', None)
    if content:
        half = len(content) // 2
        for part in (content[:half], content[half:]):
            if part:
                yield chunk(delta(content=part))
    for i, tc in enumerate(getattr(message, 'tool_calls', None) or []):
        args = tc.function.arguments or ''
        half = len(args) // 2
        yield chunk(delta(tool_calls=[SimpleNamespace(
            index=i, id=tc.id, type='function',
            function=SimpleNamespace(name=tc.function.name, arguments=args[:half]),
        )]))
        yield chunk(delta(tool_calls=[SimpleNamespace(
            index=i, id=None, type=None,
            function=SimpleNamespace(name=None, arguments=args[half:]),
        )]))
    yield chunk(delta(), finish_reason=getattr(choice, 'finish_reason', None))
    yield chunk(usage=getattr(response, 'usage', None), choices=False)
