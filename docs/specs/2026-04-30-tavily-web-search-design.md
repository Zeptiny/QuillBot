# Tavily Web Search Integration

## Summary

Replace OpenRouter's built-in `openrouter:web_search` tool with direct Tavily API calls. When web search is enabled, the bot calls Tavily before the LLM, injects results as context into the system prompt, and lets the LLM synthesize an answer with citations.

## Motivation

OpenRouter's web search tool is opaque — the bot has no control over when or how search happens, and it only works with models that support tool calling. Tavily gives direct control over search quality, result formatting, and cost, and works with any model.

## Approach

**Pre-search injection (Approach A):** Call Tavily search before the LLM, format top results into a context block appended to the system prompt, then call OpenRouter without any tools.

This is simpler than tool-calling (Approach B) because it avoids multi-turn agentic loops and model-compatibility concerns. The cost of an extra Tavily call per `/chat` is negligible (free tier: 1000 searches/month; paid: $0.004/search).

## Scope

- Affects `/chat` command and follow-up reply conversations only
- Does **not** change `/analyze`, `/spark`, `/docs`, or other commands
- OpenRouter `web_search` tool is fully removed (no fallback)

## Changes

### 1. Add `tavily-python` dependency

**File: `requirements.txt`**

Add `tavily-python>=0.5.0`.

### 2. Add `TAVILY_API_KEY` config

**File: `config.py`**

```python
TAVILY_API_KEY: Final[str | None] = os.getenv('TAVILY_API_KEY')
```

**File: `.env.example`**

Add `TAVILY_API_KEY=` line.

### 3. Add `cogs/web_search.py` module

A thin async wrapper around Tavily's search API:

- `TavilySearch` class initialized with `TAVILY_API_KEY`
- `async def search(self, query: str, max_results: int = 5) -> list[dict]` method
- Uses `tavily-python`'s `TavilyClient.search()` with `search_depth="basic"`, `topic="general"`
- Returns a list of dicts with keys: `title`, `url`, `content` (snippet)
- On any error, logs the exception and returns an empty list (never blocks the caller)

### 4. Modify `Commands` cog

**File: `cogs/commands.py`**

In `__init__`:
- Instantiate `TavilySearch` if `TAVILY_API_KEY` is set; store as `self._tavily`.

In `_run_chat()`:
- When `WEB_SEARCH_ENABLED` and `self._tavily` exist, call `await self._tavily.search(question)` before the LLM call.
- Format results into a `<web_search_results>` block appended to the system prompt:
  ```
  <web_search_results>
  [1] Title — URL
  Content snippet

  [2] Title — URL
  Content snippet
  </web_search_results>
  ```
- Remove the `tools` parameter from the `client.chat.completions.create()` call entirely (no more `openrouter:web_search`).

Update `_WEB_SEARCH_INSTRUCTIONS` to instruct the LLM to cite sources by title and URL from the injected context.

### 5. Remove OpenRouter web search tool

Delete the `tools` construction block in `_run_chat()` (lines 442-445):

```python
tools = [{
    'type': 'openrouter:web_search',
    'parameters': {'max_results': 5, 'search_context_size': 'medium'},
}] if WEB_SEARCH_ENABLED else None
```

And the `tools=tools` kwarg from the `create()` call. The LLM call becomes:

```python
response = await self.client.chat.completions.create(
    model=CHAT_MODEL,
    messages=messages,
    max_tokens=2048,
)
```

## Error handling

- If Tavily API fails (network error, rate limit, invalid key), log the exception and proceed without search context. The user still gets an LLM response, just without web data.
- If `TAVILY_API_KEY` is not set but `WEB_SEARCH_ENABLED` is `true`, search is silently skipped (no error shown to the user).

## Search context formatting

The injected block uses a simple numbered format for reliable citation by the LLM:

```
<web_search_results>
[1] PaperMC Performance Guide — https://docs.papermc.io/paper/admin/performance
PaperMC's official guide covering chunk loading, entity limits, and tick optimization for Minecraft servers.

[2] Aikar's Flags — https://mcflags.emc.gs
Recommended JVM flags for Minecraft server optimization with G1GC configuration.
</web_search_results>
```

The system prompt instructs the LLM to reference sources by number (e.g., "According to [1]...") and include the title and URL inline when citing.

## Files summary

| File | Action |
|------|--------|
| `requirements.txt` | Add `tavily-python>=0.5.0` |
| `.env.example` | Add `TAVILY_API_KEY=` |
| `config.py` | Add `TAVILY_API_KEY` env var |
| `cogs/web_search.py` | New — Tavily search wrapper |
| `cogs/commands.py` | Pre-search in `_run_chat()`, remove OpenRouter tool |