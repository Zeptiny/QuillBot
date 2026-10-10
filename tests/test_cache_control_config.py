"""LLM_CACHE_CONTROL: which providers get explicit cache_control breakpoints.

`auto` enables them only for OpenRouter and the Merge Gateway, which forward
them to Anthropic/Gemini; strict OpenAI-compatible APIs such as Mistral reject
the unknown field with a 422. An explicit true/false overrides the base URL.
"""

from config import _cache_control_enabled

from helpers import check


def test_auto_enables_only_known_gateways():
    check('openrouter', _cache_control_enabled('auto', 'https://openrouter.ai/api/v1'))
    check('merge', _cache_control_enabled('auto', 'https://api-gateway.merge.dev/v1/openai'))
    check('mistral off', not _cache_control_enabled('auto', 'https://api.mistral.ai/v1'))
    check('ollama off', not _cache_control_enabled('auto', 'http://localhost:11434/v1'))
    check('empty means auto', not _cache_control_enabled('', 'https://api.mistral.ai/v1'))


def test_explicit_setting_overrides_base_url():
    check('forced on', _cache_control_enabled('true', 'http://localhost:4000/v1'))
    check('forced on, mixed case', _cache_control_enabled(' YES ', 'http://localhost:4000/v1'))
    check('forced off', not _cache_control_enabled('false', 'https://openrouter.ai/api/v1'))
