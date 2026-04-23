# QuillBot — Audit Review: LLM, Prompting & Feature Roadmap

Items identified during the architectural audit that are **not yet implemented**.
Organized by priority.

---

## 1. LLM / Prompting Improvements

### 1.3 New Tools to Expose

| Tool | Source | Purpose |
|---|---|---|
| `check_server_status` | mcsrvstat.us API | Let the LLM check if a user's server is online when diagnosing |
| `get_paper_config_docs` | RAG + PaperMC docs | Dedicated config key lookup |
| `check_plugin_compatibility` | Modrinth version API | Check specific MC version support |
| `get_java_version_info` | Static mapping | MC version → recommended Java version |

### 1.4 Context Window Management

**Current problems:**
- `_truncate_safe(result_text, limit=6000)` per tool × `MAX_TOOL_ROUNDS=4` = up to 24K chars of tool results
- History replays 3 full Q&A pairs verbatim
- `max_tokens=1024` truncates complex answers

**Recommendations:**
```python
# Track cumulative context size
MAX_CONTEXT_CHARS = 20000

# Summarize older history
if history and len(history) > 1:
    summary = f"Resumo: {history[-2]['question']} → {history[-2]['answer'][:200]}"
    messages.append({'role': 'user', 'content': summary})
    # Only keep last exchange verbatim
    messages.append({'role': 'user', 'content': history[-1]['question']})
    messages.append({'role': 'assistant', 'content': history[-1]['answer']})

# Increase max_tokens
max_tokens=2048
```

### 1.5 De-duplicate Tool Results Across Rounds

**Current behavior:**
- If the LLM calls `search_docs` multiple times (in different tool rounds), the same document chunks may appear in both results
- The assistant repeats information redundantly in its final answer

**Action items:**
1. **Verify** — check `docs_rag.py` agentic loop to confirm if duplicate `search_docs` calls occur in practice
2. **If confirmed** — track `all_sources` across rounds and filter duplicates before sending to LLM
3. **Optional enhancement** — add a memo/context field to the LLM stating "avoid repeating these sources you just retrieved"

**Note:** This is lower priority than 1.3/1.4 and depends on runtime observation.

### 1.6 Add Gamemode Knowledge Base

**Rationale:**
- Most common admin questions involve choosing between Survival, Creative, Adventure, and Spectator
- Resource usage, difficulty balancing, and version compatibility vary per gamemode
- Currently the LLM has no structured gamemode reference

**Implementation:**
1. Create a new RAG vector store or static context chunk: `gamemode_reference.md`
   - Survival: description, typical configs, RAM usage, common settings changes
   - Creative: building/testing specific requirements, performance tips
   - Adventure: map design constraints, custom rules
   - Spectator: spectator-only server considerations
2. Include per-gamemode recommendations:
   - Most common MC versions (e.g., Survival = latest; Creative = stable)
   - PaperMC config tuning (e.g., `mob-spawning.per-player`, `difficulty`)
   - Plugin compatibility by gamemode
3. Embed during initial `/reindex` so the LLM can cite it naturally

---

## 2. Feature Roadmap

### 2.1 Interactive Timings/Spark Analysis (High Impact)

Add `/timings` command:
- Accept Spark report URL (`spark.lucko.me`)
- Parse heap summary, tick stats, top plugins from the Spark JSON API
- Feed structured data to the LLM for diagnosis
- Most commonly shared artifact in MC admin channels

### 2.2 Server Configuration Wizard (Medium Impact)

Multi-step `discord.ui.Modal`:
1. Collect: RAM, player count, MC version, server type (Paper/Purpur/Fabric)
2. LLM generates optimized config snippets from RAG context
3. Outputs `server.properties`, `paper-global.yml`, `purpur.yml` changes
4. Converts passive documentation into active, personalized guidance

### 2.3 Plugin Compatibility Checker (Medium Impact)

`/compat <plugin> <version>`:
- Query Modrinth/Hangar version-specific API endpoints
- Cross-reference with known incompatibilities
- Expose as an LLM tool (`check_compatibility`) for natural-language queries
- Addresses the most repetitive question type in support channels

---

## 3. Unimplemented Improvements from Architectural Audit

Items identified and scoped but **not yet implemented**. Ordered by priority.

### 3.1 `/spark` Command — Profiling Report Analysis ⚡ High

**Problem:** Spark is indexed as a doc source but there is no command to fetch and interpret a live Spark report. It is the single most shared performance artifact in this community.

**Value:** A user pastes `https://spark.lucko.me/abcdef` and the bot retrieves the JSON summary, interprets the top methods and ticks, and gives optimization recommendations grounded in the already-indexed Spark docs.

**Implementation idea:**
- New `@app_commands.command(name='spark')` in `cogs/log_analyzer.py` or a new cog.
- Fetch `https://spark.lucko.me/<id>.json` via `LogAnalyzer.session`.
- Extract top sampler nodes (by CPU time) and metadata (Minecraft version, server type, TPS).
- Pass a structured summary to `_run_agent` with a Spark-specific system prompt, sourcing the `"Spark"` doc source via `source_filter`.

---

### 3.2 GitHub Token Support for Indexing ⚡ High

**Problem:** All GitHub API calls during indexing (tree discovery, commit SHAs) are unauthenticated. GitHub's unauthenticated rate limit is **60 req/hour**. A single full reindex hits the tree API once per source (4 calls) plus one commit-SHA fetch per source (4 more). Exhausting the rate limit causes silent indexing failures.

**Value:** Prevents production indexing failures. Also unlocks private repos as future doc sources.

**Implementation:**
1. Add to `config.py`:
   ```python
   GITHUB_TOKEN: Final[str | None] = os.getenv('GITHUB_TOKEN')
   ```
2. In `DocsRAG.cog_load`, if `GITHUB_TOKEN` is set, pass `Authorization: Bearer <token>` as a default header on the `aiohttp.ClientSession`.
3. Document in `SETUP.md` — token only needs `public_repo` (read) scope.

---

### 3.3 Support More Paste Services ⚡ Medium

**Problem:** Only `mclo.gs` and `pastebin.com` are passively detected. `hastebin.com`, `paste.gg`, and GitHub Gist raw URLs are common in Minecraft admin communities and are silently ignored.

**Implementation:** Extend `_parse_link()` in `cogs/log_analyzer.py`:
- `hastebin.com/(\w+)` → raw: `hastebin.com/raw/\1`
- `paste.gg/p/\w+/(\w+)` → raw: `paste.gg/p/<...>/files/\1/raw`
- `gist.github.com/\w+/(\w+)` → raw: `gist.githubusercontent.com/\w+/\1/raw`

---

### 3.4 Response Feedback Buttons ⚡ Medium

**Problem:** There is no feedback signal to detect bad AI responses. Users receive incorrect answers with no recourse beyond retrying, and operators have no visibility into failure modes.

**Value:** Surfaces systematic failures (hallucinated config keys, wrong versions) and builds user trust through visible accountability.

**Implementation:**
- Add a `discord.ui.View` with 👍 / 👎 buttons to `/ask` and `/analyze` responses.
- On interaction, log `(user_id, question, answer, rating, timestamp)` to a configurable feedback channel (Discord webhook or dedicated `#bot-feedback` channel).
- No database required; structured logging is sufficient for triage.

---

## 4. Known Technical Debt

### 4.1 `OPENROUTER_API_KEY` Not Validated at Startup ⚠️ Medium

`validate_config()` in `config.py` only asserts `BOT_TOKEN`. If `OPENROUTER_API_KEY` is missing, all AI commands silently respond with "⚠️ Comando indisponível" with no startup warning, making misconfiguration hard to diagnose in production.

**Fix:** Add a warning log in `validate_config()` (or in `setup()` of AI cogs) if the key is absent:
```python
if not OPENROUTER_API_KEY:
    logger.critical("OPENROUTER_API_KEY is not set — all AI commands will be disabled")
```

### 4.2 No Exponential Backoff on LLM / HTTP Calls ⚠️ Medium

Only `RateLimitError` is caught explicitly on LLM calls. `asyncio.TimeoutError`, `aiohttp.ServerConnectionError`, and other transient failures return an error immediately with no retry. Under brief API instability the bot fails every request.

**Fix:** Wrap LLM calls in a simple retry helper with exponential backoff (2–3 attempts, base delay 1s):
```python
async def _llm_call_with_retry(self, **kwargs):
    for attempt in range(3):
        try:
            return await self.client.chat.completions.create(**kwargs)
        except (RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)
```

### 4.3 Embed Title Truncated for Long Questions ⚠️ Low

`/ask` and `/chat` set the embed title to the raw user question (`f'❓ {question}'`). Discord embed titles are capped at 256 characters and truncate silently, which looks broken for long questions.

**Fix:** Truncate the question before embedding it in the title:
```python
title_text = question if len(question) <= 240 else question[:237] + '…'
title = f'❓ {title_text}' if i == 0 else ''
```

