"""Merge Gateway vendor priority: config helper and create_chat_completion injection.

When OPENAI_BASE_URL points at merge.dev, chat completions carry an inline
`priority_order` (docs.merge.dev/merge-gateway/routing/using-policies) so the
gateway tries the configured vendors in order. `IS_MERGE_GATEWAY` and
`MERGE_PRIORITY_ORDER` are monkeypatched per test so the suite does not depend
on the developer's .env.
"""

from types import SimpleNamespace

import config
from cogs.utils import create_chat_completion
from config import chat_priority_extra_body

from helpers import as_stream, check


def test_disabled_when_base_url_is_not_merge(monkeypatch):
    monkeypatch.setattr(config, 'IS_MERGE_GATEWAY', False)
    monkeypatch.setattr(config, 'MERGE_PRIORITY_ORDER', ('particle',))
    check('no extra_body off merge.dev', chat_priority_extra_body('m/x') is None)


def test_disabled_when_priority_order_empty(monkeypatch):
    monkeypatch.setattr(config, 'IS_MERGE_GATEWAY', True)
    monkeypatch.setattr(config, 'MERGE_PRIORITY_ORDER', ())
    check('no extra_body when order empty', chat_priority_extra_body('m/x') is None)


def test_payload_uses_configured_vendor_order(monkeypatch):
    monkeypatch.setattr(config, 'IS_MERGE_GATEWAY', True)
    monkeypatch.setattr(
        config, 'MERGE_PRIORITY_ORDER',
        ('particle', 'wafer', 'fireworks', 'modal', 'zai'),
    )
    body = chat_priority_extra_body('qwen/qwen3.6-plus')
    check('priority_order set', isinstance(body, dict) and list(body) == ['priority_order'], body)
    entry = body['priority_order'][0]
    # The gateway rejects a top-level model that differs from the first entry,
    # so the entry must carry the exact model being requested.
    check('entry model equals request model', entry['model'] == 'qwen/qwen3.6-plus', entry)
    check('vendors in configured order',
          entry['vendors'] == ['particle', 'wafer', 'fireworks', 'modal', 'zai'], entry)


async def test_funnel_attaches_extra_body_on_merge(monkeypatch):
    monkeypatch.setattr(config, 'IS_MERGE_GATEWAY', True)
    monkeypatch.setattr(config, 'MERGE_PRIORITY_ORDER', ('particle', 'zai'))
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return as_stream(SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(role='assistant', content='ok', tool_calls=None),
                    finish_reason='stop',
                )],
                usage=None,
            ))

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    response = await create_chat_completion(
        client, model='m/x', messages=[{'role': 'user', 'content': 'q'}],
    )
    check('answer still produced', response.choices[0].message.content == 'ok')
    check('extra_body attached',
          captured.get('extra_body') == {
              'priority_order': [{'model': 'm/x', 'vendors': ['particle', 'zai']}],
          },
          captured.get('extra_body'))
    check('request still streamed', captured.get('stream') is True)


async def test_funnel_omits_extra_body_off_merge(monkeypatch):
    monkeypatch.setattr(config, 'IS_MERGE_GATEWAY', False)
    monkeypatch.setattr(config, 'MERGE_PRIORITY_ORDER', ('particle',))
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return as_stream(SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(role='assistant', content='ok', tool_calls=None),
                    finish_reason='stop',
                )],
                usage=None,
            ))

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    await create_chat_completion(client, model='m/x', messages=[{'role': 'user', 'content': 'q'}])
    check('no extra_body key sent', 'extra_body' not in captured, captured.get('extra_body'))
