# QuillBot — Audit Review: LLM, Prompting & Feature Roadmap

Items identified during the architectural audit that are **not yet implemented**.
Organized by priority.

> **Model defaults (last updated 2026-04-23):**
> - `CHAT_MODEL` default changed to `qwen/qwen3-6b-plus` (was `google/gemini-2.5-flash-lite`)
> - `SPARK_MODEL` added — defaults to `google/gemini-2.5-pro`; used only during active Spark sessions
>   (Spark analysis requires correlated multi-signal reasoning that exceeds flash-lite capability)

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

### 1.5 De-duplicate Tool Results Across Rounds ✅ Implemented

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

**Spark Profiler** — ✅ Implemented. See `cogs/spark.py` + `cogs/spark_parser.py`. Passive URL detection, `/spark` command, agentic detail sections, and dedicated `SPARK_MODEL` are all live.

**Legacy Timings** — Remaining work:
- Add `/timings` command accepting Paper timings report URL (`timings.aikar.co`)
- Parse the timings JSON: server info, tick breakdown, plugin timings, world stats
- Feed structured data to the LLM for diagnosis (similar to Spark flow)
- Still commonly used on older Paper servers that haven't migrated to Spark

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

### 2.4 `/java` — Java Version Recommender (Low Effort, High Use)

Standalone command exposing the `get_java_version_info` tool (§1.3) as a user-facing slash command.

- Input: MC version (e.g., `1.21.4`)
- Output: recommended Java version with explanation
- Static mapping, no external API needed:
  - 1.21+ → Java 21
  - 1.17–1.20.x → Java 17
  - ≤1.16 → Java 8
- Also useful as a quick-reference embed in support channels

### 2.5 `/compare <plugin1> <plugin2>` — Plugin Comparison (Medium Impact)

When users ask "which is better, X or Y?", fetch metadata from both plugins and present a side-by-side comparison embed:
- Downloads, last updated date, supported MC versions, description
- Uses existing `plugin_apis.py` infrastructure (Modrinth, Hangar, SpigotMC)
- Addresses a common question pattern in support channels

### 2.6 `/optimize` — Quick Optimization Checklist (Medium Impact)

Input: server type (Paper/Purpur/Spigot) + MC version.
Output: actionable checklist of recommended config optimizations.

- Paper: `chunk-loading`, `entity-activation-range`, `mob-spawning.per-player`, `view-distance`, `simulation-distance`
- Purpur: additional Purpur-specific tweaks (`purpur.yml`)
- Each item includes the config key, recommended value, and a one-line explanation
- More actionable than requiring users to know what to ask in `/ask`

### 2.7 `/serverprops <key>` — server.properties Quick Reference (Low Effort)

Lookup any `server.properties` key and return:
- Description of what it does
- Default value and valid range
- Common tuning advice
- Backed by RAG over PaperMC docs or a static knowledge base
- Related to §1.3 `get_paper_config_docs` tool but as a user-facing command

### 2.8 `/health` — Bot Diagnostics (Low Effort, Operator-Facing) ✅ Implemented

Admin-only command showing bot health status:
- API connectivity: OpenRouter, GitHub, mclo.gs, Modrinth (latency + status)
- Vector store: age, chunk count, last reindex timestamp
- Conversation cache: active conversations, TTL stats
- Error counts since startup (by category)
- Useful for operators to verify bot health after deployment or config changes

---

## 3. Unimplemented Improvements from Architectural Audit

Items identified and scoped but **not yet implemented**. Ordered by priority.

### 3.1 GitHub Token Support for Indexing ⚡ High

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

### 3.2 Support More Paste Services ⚡ Medium

**Problem:** Only `mclo.gs` and `pastebin.com` are passively detected. `hastebin.com`, `paste.gg`, and GitHub Gist raw URLs are common in Minecraft admin communities and are silently ignored.

**Implementation:** Extend `_parse_link()` in `cogs/log_analyzer.py`:
- `hastebin.com/(\w+)` → raw: `hastebin.com/raw/\1`
- `paste.gg/p/\w+/(\w+)` → raw: `paste.gg/p/<...>/files/\1/raw`
- `gist.github.com/\w+/(\w+)` → raw: `gist.githubusercontent.com/\w+/\1/raw`

---

### 3.3 Spark Tool Set — Intentionally Omitted Tools ℹ️

During the Spark integration design (see `SPARK_INTEGRATION.md §14`), two tools were explicitly excluded from the Spark agent's tool set:

**`web_search` — Removed from Spark sessions.**
Uncontrolled web results introduce noise and hallucination risk in a structured diagnostic workflow. The indexed PaperMC + Spark RAG docs provide grounded, version-specific recommendations without external noise. `web_search` remains available in `/ask` and `/chat` commands where open-ended Q&A benefits from real-time information.

**`get_known_issues` — Not implemented.**
This tool was considered to surface known plugin performance bugs. Without `web_search` as a data source, it has no reliable runtime data. Plugin-specific issues should be surfaced by `search_docs(source="PaperMC")` and by directing users to the plugin's own issue tracker.

---

### 3.4 Response Feedback Buttons ⚡ Medium

**Problem:** There is no feedback signal to detect bad AI responses. Users receive incorrect answers with no recourse beyond retrying, and operators have no visibility into failure modes.

**Value:** Surfaces systematic failures (hallucinated config keys, wrong versions) and builds user trust through visible accountability.

**Implementation:**
- Add a `discord.ui.View` with 👍 / 👎 buttons to `/ask` and `/analyze` responses.
- On interaction, log `(user_id, question, answer, rating, timestamp)` to a configurable feedback channel (Discord webhook or dedicated `#bot-feedback` channel).
- No database required; structured logging is sufficient for triage.

---

### 3.5 Passive Crash Report Detection ⚡ Medium

**Problem:** Crash reports shared as file attachments (`.txt`, `.log`) are not auto-detected. Users must manually run `/analyze` after uploading a crash report, unlike mclo.gs/pastebin links which trigger automatic pattern matching.

**Implementation:**
- Extend `on_message` in `cogs/log_analyzer.py` to check file attachments for `---- Minecraft Crash Report ----`
- Auto-trigger analysis (same flow as mclo.gs link detection)
- Add `👀` reaction and offer "Analyze with AI" button
- Crash reports are one of the most frequently shared artifacts in support channels

---

### 3.6 Per-Channel Feature Configuration ⚡ Medium

**Problem:** All bot features are globally active. Server admins cannot disable passive log detection in off-topic channels or restrict `/ask` to specific support channels.

**Implementation:**
- Admin command: `/config <feature> <channel> enable|disable`
- Features: `log_detection`, `spark_detection`, `passive_errors`, `ask`, `analyze`
- Store configuration in `data/channel_config.json`
- Check config in each cog's `on_message` listener and command checks
- Default: all features enabled in all channels (backward compatible)

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

### 4.4 Logging Framework Improvements ⚠️ Low ✅ Implemented

`main.py` already configures `logging.basicConfig()` with a proper format, but some cogs may still use `print()` for output. Additionally, log messages lack structured context (guild_id, user_id, command name), making production debugging harder.

**Fixes:**
1. Audit all cogs for stray `print()` calls → replace with `logger.info/warning/error`
2. Add structured context to log messages where relevant:
   ```python
   logger.info("Processing /ask from user=%s guild=%s", interaction.user.id, interaction.guild_id)
   ```
3. Add `LOG_LEVEL` env var to `config.py` (default: `INFO`), apply in `main.py`:
   ```python
   logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), ...)
   ```

---

## 5. Error Pattern Coverage Expansion

The current 18 patterns in `responses/errors.py` cover common cases. The following patterns are missing and would improve passive detection coverage:

| Pattern | Category | Notes |
|---|---|---|
| `OutOfMemoryError: Metaspace` | Memory | Different from heap OOM — indicates too many plugins loaded |
| `ConcurrentModificationException` | World | Usually from async world access or corrupted chunks |
| `java.lang.StackOverflowError` | Performance | Recursive chunk loading or plugin logic loops |
| `ModResolutionException` | Forge/Fabric | Mod dependency resolution failures |
| `IncompatibleModSetException` | Forge/Fabric | Mod version conflicts |
| `ExceptionInInitializerError` | Plugin | Plugin static initialization failures |
| `BackendConnectionException` | Proxy | BungeeCord/Velocity backend connection failures |
| `io.netty.handler.codec.DecoderException` | Network | Packet decoding errors, often from version mismatch |

**Implementation:** Add entries to `_raw_patterns` in `responses/errors.py` with pt-BR response templates. Follow existing ordering convention: specific patterns before generic ones within each category.

---

## 6. Testing & Infrastructure

### 6.1 Proper Test Suite ⚡ High

**Problem:** The only test file (`tests/test_spark_parser.py`, 954 lines) duplicates the entire Spark parser and has no assertions, fixtures, or test runner. There is no pytest in `requirements.txt`.

**Implementation:**
1. Add `pytest` and `pytest-asyncio` to `requirements.txt`
2. Refactor `tests/test_spark_parser.py` into proper pytest tests with assertions
3. Add tests for:
   - `cogs/utils.py` — `truncate_safe` and `split_response` edge cases (empty string, exact boundary, unicode)
   - `responses/errors.py` — pattern matching against known log snippets
   - `cogs/plugin_apis.py` — URL construction and response parsing with mock data
   - `cogs/spark_parser.py` — `build_summary()` and `build_detail()` with fixture JSON
4. Add a `conftest.py` with shared fixtures (mock Spark JSON, sample logs)

### 6.2 Docker Support ⚡ Medium ✅ Implemented

**Problem:** No containerization. Deployment requires manual Python setup, which is a barrier for non-Python operators.

**Implementation:**
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["python", "main.py"]
   ```
2. Create `docker-compose.yml` with volume mount for `data/` (vector store persistence) and `.env` passthrough
3. Document in `SETUP.md`

### 6.3 CI/CD Pipeline ⚡ Low

Add a GitHub Actions workflow:
1. **Lint** — `ruff check .` or `flake8`
2. **Test** — `pytest` (once 6.1 is implemented)
3. **Build** — verify `pip install -r requirements.txt` succeeds
4. Trigger on push to `main` and on PRs
5. Optional: automated Docker image build and push to GHCR

