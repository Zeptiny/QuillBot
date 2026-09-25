"""run_tool_loop error handling: a failing tool is reported to the model instead of aborting the turn."""

import asyncio
from types import SimpleNamespace

from cogs.utils import run_tool_loop

from helpers import check


class _Completions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append([dict(m) if isinstance(m, dict) else m for m in kwargs['messages']])
        return self._responses.pop(0)


def _response(content=None, tool_calls=None):
    tcs = [
        SimpleNamespace(id=f'call_{i}', type='function',
                        function=SimpleNamespace(name=name, arguments=args))
        for i, (name, args) in enumerate(tool_calls or [])
    ] or None
    message = SimpleNamespace(role='assistant', content=content, tool_calls=tcs)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')], usage=None)


def test_tool_exception_reported_to_model():
    """tool exception does not abort the turn."""
    completions = _Completions([
        _response(tool_calls=[('search_docs', '{"max_results": "cinco"}')]),
        _response('resposta final'),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def exec_tool(name, args):
        int(args['max_results'])  # ValueError, like a malformed model argument
        return 'unreachable', []

    answer, sources, _ = asyncio.run(run_tool_loop(
        client, 'model-x', [{'role': 'user', 'content': 'q'}], [], exec_tool,
    ))
    check('answer still produced', answer == 'resposta final', answer)
    check('no sources', sources == [])
    tool_msgs = [m for m in completions.calls[1] if isinstance(m, dict) and m.get('role') == 'tool']
    check('error surfaced as tool result',
          bool(tool_msgs) and 'Erro ao executar a ferramenta search_docs' in tool_msgs[0]['content'],
          tool_msgs)
